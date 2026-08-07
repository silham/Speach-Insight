import React from 'react';
import { Dashboard } from './pages/Dashboard';
import { Login } from './pages/Login';
import { AuthProvider } from './hooks/useAuth';
import { useAuth } from './hooks/authContext';
import './App.css';

const Gate = () => {
  const { user, checking } = useAuth();

  if (checking) {
    return <div className="auth-checking">Loading…</div>;
  }
  return user ? <Dashboard /> : <Login />;
};

function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}

export default App;
