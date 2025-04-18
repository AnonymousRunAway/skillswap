// src/components/common/Header.jsx
import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  Button, 
  IconButton, 
  Avatar, 
  Menu, 
  MenuItem, 
  Box 
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';

const Header = () => {
  const { currentUser, logout } = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState(null);
  
  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  const handleLogout = () => {
    logout();
    handleMenuClose();
    navigate('/login');
  };

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar>
        <Typography
          variant="h6"
          component={Link}
          to="/"
          sx={{ 
            flexGrow: 1, 
            textDecoration: 'none', 
            color: 'inherit',
            fontWeight: 'bold'
          }}
        >
          <span style={{ color: '#1dbf73' }}>Skill</span>Swap
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {currentUser ? (
            <>
              <Button 
                component={Link} 
                to="/matches" 
                color="inherit" 
                sx={{ mr: 2 }}
              >
                Matches
              </Button>
              
              <Button 
                component={Link} 
                to="/transactions" 
                color="inherit" 
                sx={{ mr: 2 }}
              >
                Transactions
              </Button>
              
              <IconButton onClick={handleMenuOpen}>
                <Avatar 
                  src={currentUser.image_path} 
                  alt={currentUser.username}
                  sx={{ width: 32, height: 32 }}
                />
              </IconButton>
              
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                <MenuItem 
                  onClick={() => {
                    handleMenuClose();
                    navigate('/profile');
                  }}
                >
                  Profile
                </MenuItem>
                <MenuItem onClick={handleLogout}>Logout</MenuItem>
              </Menu>
            </>
          ) : (
            <>
              <Button 
                component={Link} 
                to="/login" 
                color="inherit" 
                sx={{ mr: 2 }}
              >
                Login
              </Button>
              
              <Button 
                component={Link} 
                to="/register" 
                variant="contained" 
                color="primary"
              >
                Register
              </Button>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Header;
