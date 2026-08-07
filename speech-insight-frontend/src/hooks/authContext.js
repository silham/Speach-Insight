import { createContext, useContext } from 'react';

// Kept in its own module (no component exports) so React Fast Refresh keeps
// working for the provider component.
export const AuthContext = createContext(null);

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
};
