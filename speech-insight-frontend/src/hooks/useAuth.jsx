import React, { useEffect, useState, useCallback } from 'react';
import { api, setToken, clearToken, getToken } from '../api/client';
import { AuthContext } from './authContext';

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  // Only "checking" when there is actually a token to validate — otherwise we
  // already know nobody is signed in and can render the login screen at once.
  const [checking, setChecking] = useState(() => Boolean(getToken()));

  useEffect(() => {
    if (!getToken()) return;

    api.get('/auth/me')
      .then((res) => setUser(res.data))
      .catch(() => clearToken())
      .finally(() => setChecking(false));
  }, []);

  // The axios interceptor fires this when any call comes back 401.
  useEffect(() => {
    const onUnauthorized = () => setUser(null);
    window.addEventListener('si-unauthorized', onUnauthorized);
    return () => window.removeEventListener('si-unauthorized', onUnauthorized);
  }, []);

  const login = useCallback(async (email, password) => {
    const res = await api.post('/auth/login', { email, password });
    setToken(res.data.access_token);
    setUser(res.data.user);
    return res.data.user;
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
  }, []);

  const value = { user, checking, login, logout, isAdmin: user?.role === 'admin' };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
