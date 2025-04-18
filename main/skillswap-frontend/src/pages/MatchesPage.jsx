// src/pages/MatchesPage.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, 
  Typography, 
  Box, 
  TextField, 
  InputAdornment, 
  CircularProgress, 
  Grid,
  Button
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import MatchCard from '../components/matches/MatchCard';

const MatchesPage = () => {
  const { currentUser, refreshUser } = useAuth();
  const navigate = useNavigate();
  const [matches, setMatches] = useState([]);
  const [matchesLoading, setMatchesLoading] = useState(true);
  const [matchesError, setMatchesError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);

  const handleTransactionCreated = () => {
    refreshUser();
  };

  // Debug-enhanced search function
  const handleSearch = async () => {
    console.log('Starting search with query:', searchQuery);
    
    if (!searchQuery.trim()) {
      setSearchError('Please enter a search term');
      return;
    }

    if (!currentUser?.user_id) {
      setSearchError('Please login to search');
      return;
    }

    try {
      setSearchLoading(true);
      setSearchError(null);
      setSearchResults([]);
      
      console.log('Making API call to searchSkills...');
      const response = await api.searchSkills(currentUser.user_id, searchQuery);
      console.log('API response:', response);
      
      // Debug the response structure
      if (!response) {
        console.error('API returned undefined response');
        throw new Error('No response from server');
      }
      
      console.log('Response data:', response.data);
      
      const results = Array.isArray(response?.data) ? response.data : [];
      console.log('Processed results:', results);
      
      setSearchResults(results);
      
      if (results.length === 0) {
        console.log('No results found for query');
        setSearchError('No matches found for your search');
      }
    } catch (err) {
      console.error('Search error:', err);
      console.error('Error details:', {
        message: err.message,
        response: err.response,
        stack: err.stack
      });
      setSearchError(err.message || 'Search failed. Please try again.');
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
      console.log('Search completed. Current searchResults:', searchResults);
    }
  };

  // Rest of your component remains the same...
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        {/* Search Section - Enhanced with debug info */}
        <Box sx={{ mb: 4 }}>
          <TextField
            fullWidth
            placeholder="Search for skills..."
            value={searchQuery}
            onChange={(e) => {
              console.log('Search query changed:', e.target.value);
              setSearchQuery(e.target.value);
            }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  <Button 
                    variant="contained" 
                    onClick={() => {
                      console.log('Search button clicked');
                      handleSearch();
                    }}
                    disabled={!searchQuery.trim() || searchLoading}
                  >
                    {searchLoading ? 'Searching...' : 'Search'}
                  </Button>
                </InputAdornment>
              )
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                console.log('Enter key pressed for search');
                handleSearch();
              }
            }}
          />
          {searchError && (
            <Typography color="error" variant="body2" sx={{ mt: 1 }}>
              {searchError}
            </Typography>
          )}
        </Box>

        {/* Debug output panel */}
        <Box sx={{ 
          p: 2, 
          mb: 4, 
          backgroundColor: '#f5f5f5', 
          borderRadius: 1 
        }}>
          <Typography variant="subtitle2">Debug Information:</Typography>
          <Typography variant="body2">
            Current Query: {searchQuery}
          </Typography>
          <Typography variant="body2">
            Results Count: {searchResults.length}
          </Typography>
          <Typography variant="body2">
            Loading: {searchLoading ? 'Yes' : 'No'}
          </Typography>
          <Typography variant="body2">
            Error: {searchError || 'None'}
          </Typography>
          <Button 
            size="small" 
            onClick={() => console.log('Current searchResults:', searchResults)}
            sx={{ mt: 1 }}
          >
            Log Results to Console
          </Button>
        </Box>

        {/* Search Results with safety checks */}
        {searchLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        )}
        
        {searchResults?.length > 0 ? (
          <Box sx={{ mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              Search Results
            </Typography>
            <Grid container spacing={3}>
              {searchResults.map((result, index) => {
                console.log(`Rendering result ${index}:`, result);
                return (
                  <Grid item xs={12} md={6} key={result?.user_id || `result-${index}`}>
                    {result ? (
                      <MatchCard 
                        match={result} 
                        onTransactionCreated={handleTransactionCreated}
                      />
                    ) : (
                      <Typography color="error">Invalid result data</Typography>
                    )}
                  </Grid>
                );
              })}
            </Grid>
          </Box>
        ) : !searchLoading && searchQuery && (
          <Typography variant="body1" color="textSecondary" sx={{ textAlign: 'center' }}>
            {searchError || 'No results found'}
          </Typography>
        )}

        {/* ... rest of your existing JSX ... */}
      </Box>
    </Container>
  );
};

export default MatchesPage;