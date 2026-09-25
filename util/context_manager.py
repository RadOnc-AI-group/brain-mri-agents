from typing import Any, Dict

class SessionContext:
    """Simple key-value registry accessible across tools."""
    def __init__(self):
        self._store: Dict[str, Any] = {}

    def set(self, key: str, value: Any):
        """Save an object by key."""
        self._store[key] = value

    def get(self, key: str, default=None) -> Any:
        """Retrieve an object by key."""
        return self._store.get(key, default)

    def delete(self, key: str):
        """Remove an object."""
        if key in self._store:
            del self._store[key]

    def list_keys(self):
        """List all stored variable names."""
        return list(self._store.keys())
    
    def __delattr__(self, name):
        self.delete(name)

    def __setitem__(self, name, value):
        self.set(name, value)
    
    def __getitem__(self, name):
        if name in self._store:
            return self._store[name]
        raise AttributeError(f"'SessionContext' object has no attribute '{name}'")

    def __len__(self):
        return len(self._store)
    
    def __dir__(self):
        return self.list_keys()
    
    def __contains__(self, key):
        return key in self._store
    

class PatientContext(SessionContext):
    """Context for storing patient-specific data. It will be nested within SessionContext."""
    pass
    


# session_context = SessionContext()


# if __name__ == "__main__":
#     session_context.set("a", 123)
#     print(session_context.get("a"))
    