import axios from 'axios';

export const API_BASE = 'http://127.0.0.1:8000';

const TOKEN_KEY = 'si-token';

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t);
export const clearToken = () => localStorage.removeItem(TOKEN_KEY);

export const api = axios.create({ baseURL: API_BASE });

// Attach the bearer token to every request.
api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// A 401 means the token is missing, expired, or the account is gone — drop it
// and let the app fall back to the login screen.
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      clearToken();
      window.dispatchEvent(new Event('si-unauthorized'));
    }
    return Promise.reject(err);
  }
);

/** Absolute URL for a segment audio clip served by the backend. */
export const audioUrl = (url) => (url.startsWith('http') ? url : `${API_BASE}${url}`);
