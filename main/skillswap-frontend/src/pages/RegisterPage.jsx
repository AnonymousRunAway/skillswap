// src/pages/RegisterPage.jsx
import React from 'react';
import { Container } from '@mui/material';
import Register from '../components/auth/Register';

const RegisterPage = () => {
  return (
    <Container maxWidth="md">
      <Register />
    </Container>
  );
};

export default RegisterPage;
