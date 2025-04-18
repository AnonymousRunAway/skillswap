// src/pages/HomePage.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Container, 
  Typography, 
  Button, 
  Grid, 
  Card, 
  CardContent, 
  Box,
  CircularProgress
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import FeedItem from '../components/feed/FeedItem';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
};

const HomePage = () => {
  const { currentUser } = useAuth();
  const navigate = useNavigate();
  const [feedItems, setFeedItems] = useState([]);
  const [loading, setLoading] = useState(true); // Initialize as true
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFeed = async () => {
      if (!currentUser?.user_id) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await api.getFeed(currentUser.user_id);
        setFeedItems(response || []); // Ensure array even if undefined
      } catch (err) {
        console.error('Feed fetch error:', err);
        setError('Failed to load feed');
        setFeedItems([]); // Reset to empty array
      } finally {
        setLoading(false);
      }
    };

    // Add small delay to ensure auth is ready
    const timer = setTimeout(fetchFeed, 100); 
    return () => clearTimeout(timer);
  }, [currentUser?.user_id]); // Only re-run if user_id changes

  // Render loading state
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  // Render error state
  if (error) {
    return (
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography color="error">{error}</Typography>
        <Button 
          variant="contained" 
          sx={{ mt: 2 }}
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </Box>
    );
  };
  
  return (
    <Container maxWidth="lg">
      {!currentUser ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ 
            textAlign: 'center', 
            py: 8,
            backgroundImage: 'linear-gradient(135deg, #1dbf73 0%, #19a463 100%)',
            color: 'white',
            borderRadius: 2,
            mb: 6,
            px: 2
          }}>
            <Typography variant="h2" component="h1" gutterBottom>
              Exchange Skills, Grow Together
            </Typography>
            <Typography variant="h5" paragraph>
              Connect with others who want to learn what you know, and teach what you want to learn.
            </Typography>
            <Button 
              variant="contained" 
              size="large"
              sx={{ 
                bgcolor: 'white', 
                color: '#1dbf73',
                '&:hover': {
                  bgcolor: '#f5f5f5',
                }
              }}
              onClick={() => navigate('/register')}
            >
              Join SkillSwap Today
            </Button>
          </Box>
          
          <Typography variant="h4" component="h2" align="center" gutterBottom>
            How It Works
          </Typography>
          
          <Grid container spacing={4} sx={{ mb: 6 }}>
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h5" component="h3" gutterBottom>
                    1. Create Your Profile
                  </Typography>
                  <Typography variant="body1">
                    Tell us what you can teach and what you want to learn.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h5" component="h3" gutterBottom>
                    2. Find Matches
                  </Typography>
                  <Typography variant="body1">
                    We'll connect you with people who are perfect skill matches.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Typography variant="h5" component="h3" gutterBottom>
                    3. Start Learning
                  </Typography>
                  <Typography variant="body1">
                    Exchange skills and grow together.
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </motion.div>
            ) : (
              <Box sx={{ py: 4 }}>
                <Typography variant="h4" component="h1" gutterBottom>
                  Welcome back, {currentUser.username}!
                </Typography>
      
                {loading ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
                    <CircularProgress />
                  </Box>
                ) : (
                  <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                  >
                    {feedItems.length > 0 ? (
                      <Grid container spacing={3}>
                        {feedItems.map((item, index) => (
                          <Grid item xs={12} key={index}>
                            <FeedItem item={item} />
                          </Grid>
                        ))}
                      </Grid>
                    ) : (
                      <Box sx={{ textAlign: 'center', my: 4 }}>
                        <Typography variant="h6" color="text.secondary">
                          No activity in your feed yet.
                        </Typography>
                        <Button
                          variant="contained"
                          color="primary"
                          sx={{ mt: 2 }}
                          onClick={() => navigate('/matches')}
                        >
                          Find Matches
                        </Button>
                      </Box>
                    )}
                  </motion.div>
                )}
              </Box>
            )}
          </Container>
        );
      };
      
      export default HomePage;
      