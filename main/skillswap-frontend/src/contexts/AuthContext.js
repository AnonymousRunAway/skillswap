// src/contexts/AuthContext.js
import React, { createContext, useState, useEffect, useContext } from 'react';
import api from '../services/api';

const AuthContext = createContext();

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Check if user is logged in from localStorage
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      setCurrentUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    try {
      setError(null);
      const response = await api.login(username, password);
      const userData = response.data;
      
      if (userData.success) {
        // Get full user profile
        const userResponse = await api.getUser(userData.user_id);
        const userProfile = userResponse.data;
        
        setCurrentUser(userProfile);
        localStorage.setItem('user', JSON.stringify(userProfile));
        return true;
      }
      return false;
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
      return false;
    }
  };

  const register = async (userData, profileImage) => {
    try {
      setError(null);
      const response = await api.register(userData, profileImage);
      return response.data;
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed');
      throw err;
    }
  };

  const logout = () => {
    localStorage.removeItem('user');
    setCurrentUser(null);
  };

  const refreshUser = async () => {
    if (currentUser) {
      try {
        const response = await api.getUser(currentUser.user_id);
        const updatedUser = response.data;
        setCurrentUser(updatedUser);
        localStorage.setItem('user', JSON.stringify(updatedUser));
      } catch (err) {
        console.error('Failed to refresh user:', err);
      }
    }
  };

  const value = {
    currentUser,
    loading,
    error,
    login,
    register,
    logout,
    refreshUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
