import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure
from skimage.measure import label, regionprops
# from skimage.morphology import ball, ellipse
from monai.transforms import FillHoles, KeepLargestConnectedComponent
from sympy import intersection
from pathlib import Path

def ellipsoid_kernel(radius_mm, spacing):
    """Create an ellipsoid structuring element based on the given radius and spacing."""
    radius_voxels = np.maximum(1, np.round(radius_mm / np.array(spacing))).astype(int)
    # Create a grid of coordinates
    x, y, z = np.ogrid[-radius_voxels[0]:radius_voxels[0]+1,
                          -radius_voxels[1]:radius_voxels[1]+1,
                          -radius_voxels[2]:radius_voxels[2]+1]
    
    dist = (x / radius_voxels[0])**2 + (y / radius_voxels[1])**2 + (z / radius_voxels[2])**2
    return dist <= 1

def expand_segmentation(segmentation, radius_mm, spacing):
    """Expand the segmentation by a specified radius in millimeters."""
    kernel = ellipsoid_kernel(radius_mm, spacing)
    expanded = binary_dilation(segmentation, structure=kernel)
    return expanded

def erode_segmentation(segmentation, radius_mm, spacing):
    """Erode the segmentation by a specified radius in millimeters."""
    kernel = ellipsoid_kernel(radius_mm, spacing)
    eroded = binary_erosion(segmentation, structure=kernel)
    return eroded

def fill_holes(segmentation):
    """Fill holes in the segmentation."""
    fill_holes_transform = FillHoles(applied_labels=[1]) 
    filled = fill_holes_transform(np.expand_dims(segmentation.astype(np.uint8), axis=0))
    return filled[0].astype(bool)

def keep_largest_connected_component(segmentation):
    """Keep only the largest connected component in the segmentation."""
    # if not np.any(segmentation):
    #     return segmentation  # Return the original if there are no components
    
    # keep_largest_transform = KeepLargestConnectedComponent(applied_labels=[1])
    # largest = keep_largest_transform(np.expand_dims(segmentation.astype(np.uint8), axis=0))
    # return largest[0].astype(bool)

    labeled = label(segmentation)
    if labeled.max() == 0:
        return segmentation  # No components found, return original
    counts = np.bincount(labeled.ravel())
    counts[0] = 0 
    largest_label = np.argmax(counts)

    return labeled == largest_label

def find_furthest_point(seg_res_cavity, expanded_seg):
    rc_coords = np.argwhere(seg_res_cavity)
    expanded_coords = np.argwhere(expanded_seg)

    if len(rc_coords) == 0 or len(expanded_coords) == 0:
        return None
    
    tree = cKDTree(expanded_coords)
    distances, _ = tree.query(rc_coords)
    furthest_idx = np.argmax(distances)
    return tuple(rc_coords[furthest_idx])

def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def find_nearest_intersection(centroid, vector, segmentation):
    vector_length = int(np.linalg.norm(vector))
    if vector_length == 0: return None

    norm_v = normalize_vector(vector)
    # raycast outward from the centroid along the vector direction
    for i in range(1, vector_length + 1):
        point = np.round(centroid + norm_v * i).astype(int)
        if (0 <= point[0] < segmentation.shape[0] and
            0 <= point[1] < segmentation.shape[1] and
            0 <= point[2] < segmentation.shape[2]):
            if segmentation[tuple(point)]:
                return point

    return None

def find_nearest_points(seg_res_cavity, expanded_seg, centroid):
    output_data = np.zeros_like(seg_res_cavity, dtype=np.bool_)
    if centroid is None:
        return output_data
    
    seg_indices = np.argwhere(expanded_seg)
    for idx in seg_indices:
        vector = idx - centroid
        nearest_point = find_nearest_intersection(centroid, vector, expanded_seg)
        if nearest_point is not None:
            output_data[tuple(nearest_point)] = True

    return output_data

def calculate_intersection_lengths(first_layer_coords, centroid, expanded_seg, spacing):
    intersection_lengths = []

    for coord in first_layer_coords:
        vector = coord - centroid
        norm_vector = normalize_vector(vector)
        if np.all(norm_vector == 0):
            continue

        point = np.array(coord, dtype=float)
        length = 0.0

        while True:
            point += norm_vector
            idx = np.round(point).astype(int)

            if (0 <= idx[0] < expanded_seg.shape[0] and
                0 <= idx[1] < expanded_seg.shape[1] and
                0 <= idx[2] < expanded_seg.shape[2]) and expanded_seg[tuple(idx)]:

                length += np.linalg.norm(norm_vector * spacing)
            else:
                break
        intersection_lengths.append(length) 
    
    return intersection_lengths

def process_single_label(rc_seg, brain_seg, spacing, expansion_mm=5):
    # --- Part 1: Expanded Dura ---
    overlapping_points = rc_seg & brain_seg
    expanded_dura = expand_segmentation(overlapping_points, expansion_mm, spacing)  # expansion_mm margin #
    intersected_dura = expanded_dura & brain_seg

    # --- Part 2: Adapted Dura ---
    centroid = find_furthest_point(rc_seg, intersected_dura)
    if centroid is None:
        print("Warning: No centroid found, skipping label.")
        return np.zeros_like(rc_seg, dtype=bool)

    first_layer = find_nearest_points(rc_seg, intersected_dura, centroid)
    first_layer_filled = fill_holes(first_layer)
    
    first_layer_coords = np.argwhere(first_layer_filled)
    intersection_lengths = calculate_intersection_lengths(first_layer_coords, centroid, intersected_dura, spacing)
    
    if len(intersection_lengths) == 0:
        median_length = 1
    else:
        median_length = np.median(intersection_lengths)
        
    if np.isnan(median_length) or median_length <= 0:
        median_length = 1

    expanded_first_layer = expand_segmentation(first_layer_filled, median_length + 1, spacing)
    overlapping_first_layer = intersected_dura & expanded_first_layer

    # --- Part 3: Intraparenchymal ---
    seg_cavity_expanded = expand_segmentation(rc_seg, 2, spacing)
    seg_cavity_expanded_subtracted = seg_cavity_expanded & ~brain_seg
    seg_cavity_expanded_subtracted_eroded = erode_segmentation(seg_cavity_expanded_subtracted, 2, spacing)
    seg_cavity_expanded_subtracted_eroded_exp = expand_segmentation(seg_cavity_expanded_subtracted_eroded, 2, spacing)
    
    seg_cavity_keep = keep_largest_connected_component(seg_cavity_expanded_subtracted_eroded_exp)
    
    seg_cavity_keep_plusdura = seg_cavity_keep | overlapping_first_layer
    seg_cavity_keep_plusdura_filled = fill_holes(seg_cavity_keep_plusdura)
    
    final_ctv_label = seg_cavity_keep_plusdura_filled | rc_seg
    
    return final_ctv_label



def process_multiple_labels(rc_data, brain_data, spacing, expansion_mm=[5]):
    """Isolates multiple labels in the RC data and processes them independently."""
    labeled_components = label(rc_data)
    num_labels = labeled_components.max()
    
    if num_labels == 0:
        # print("Warning: No resection cavity detected.")
        return np.zeros_like(rc_data, dtype=bool)
        
    # print(f"Found {num_labels} connected components in RC. Processing...")

    if isinstance(expansion_mm, (int, float)):
        expansion_mm = [expansion_mm] * num_labels
    elif len(expansion_mm) != num_labels:
        raise ValueError("Length of expansion_mm list must match the number of labels in RC.")
    
    final_combined_result = np.zeros_like(rc_data, dtype=bool)
    
    for i in range(1, num_labels + 1):
        # print(f"Processing component {i}/{num_labels}...")
        single_label_seg = (labeled_components == i)
        
        single_result = process_single_label(single_label_seg, brain_data, spacing, expansion_mm[i-1])
        final_combined_result |= single_result  # Combine results directly using logical OR
        
    return final_combined_result




if __name__ == "__main__":
    # Example usage
    base_dir = Path("/home/cerdur/brain-mri-agents/playground")
    patient = "TUM_200"
    rc_path = base_dir / f"{patient}_rc.nii.gz"
    brain_path = base_dir / f"{patient}_dura.nii.gz"

    rc_img = nib.load(rc_path)
    brain_img = nib.load(brain_path)

    rc_data = rc_img.get_fdata().astype(bool)
    brain_data = brain_img.get_fdata().astype(bool)
    spacing = rc_img.header.get_zooms()[:3]

    final_result = process_multiple_labels(rc_data, brain_data, spacing)

    # Save the result as a new NIfTI file
    result_img = nib.Nifti1Image(final_result.astype(np.uint8), affine=rc_img.affine)
    nib.save(result_img, base_dir / f"{patient}_final_ctv_label.nii.gz")