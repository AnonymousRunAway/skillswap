// src/pages/LoginPage.jsx
import React from 'react';
import { Container } from '@mui/material';
import Login from '../components/auth/Login';

const LoginPage = () => {
  return (
    <Container maxWidth="sm">
      <Login />
    </Container>
  );
};

export default LoginPage;
