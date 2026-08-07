import React, { useState } from 'react';
import { useAuth } from '../hooks/authContext';

export const Login = () => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  // Login screen follows the saved theme so it doesn't flash a different
  // palette than the dashboard the user lands on.
  const isDark = localStorage.getItem('si-theme') !== 'light';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email || !password) return;

    setBusy(true);
    setError('');
    try {
      await login(email, password);
    } catch (err) {
      setError(
        err.response?.status === 401
          ? 'Incorrect email or password.'
          : err.response?.data?.detail || 'Could not reach the server.'
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-screen" data-theme={isDark ? 'dark' : 'light'}>
      <form className="login-card" onSubmit={handleSubmit}>
        <div className="login-brand">
          <div className="brand-icon-wrapper">
            <svg className="brand-logo-icon" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
              <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
              <line x1="12" x2="12" y1="19" y2="22" />
            </svg>
          </div>
          <div>
            <h3>SpeechInSight</h3>
            <span>Analytics Engine</span>
          </div>
        </div>

        <h4 className="login-title">Sign in</h4>
        <p className="login-subtitle">
          Accounts are provisioned by an administrator. There is no self-signup.
        </p>

        <label className="login-field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            autoComplete="username"
            autoFocus
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
        </label>

        <label className="login-field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            autoComplete="current-password"
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />
        </label>

        {error && <div className="login-error" role="alert">{error}</div>}

        <button className="btn-primary" type="submit" disabled={busy || !email || !password}>
          {busy ? 'Signing in…' : 'Sign In'}
        </button>
      </form>
    </div>
  );
};
