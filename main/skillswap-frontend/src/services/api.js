// src/services/api.js
import axios from 'axios';

// Set the base URL to your API endpoint
const API_URL = 'http://localhost:8000'; // Replace with your actual ngrok URL

const api = {
  // Auth endpoints
  register: (userData, profileImage) => {
    const formData = new FormData();
    
    // Add user data as JSON
    formData.append('user_data', JSON.stringify(userData));
    
    // Add profile image if provided
    if (profileImage) {
      formData.append('file', profileImage);
    }
    
    return axios.post(`${API_URL}/register`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  login: (username, password) => {
    return axios.post(`${API_URL}/login`, { username, password });
  },
  
  getUser: (userId) => {
    return axios.get(`${API_URL}/users/${userId}`);
  },
  
  // Matches endpoints
  getMatches: (userId) => {
    return axios.get(`${API_URL}/matches/${userId}`);
  },
  
  // Transactions endpoints
  createTransaction: (transactionData) => {
    return axios.post(`${API_URL}/transactions`, transactionData);
  },
  
  getUserTransactions: (userId) => {
    return axios.get(`${API_URL}/transactions?user_id=${userId}`);
  },
  
  confirmTransaction: (transactionId, userId, rating) => {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('rating', rating);
    
    return axios.post(`${API_URL}/transactions/${transactionId}/confirm`, formData);
  },
  
  // Feed endpoint
// ONLY REPLACE THE getFeed METHOD, KEEP EVERYTHING ELSE EXACTLY THE SAME
getFeed: async (userId) => {
    if (!userId) {
      console.error('No user ID provided to getFeed');
      return [];
    }
  
    try {
      const response = await axios.get(`${API_URL}/feed/${userId}`, {
        headers: {
          'Accept': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        }
      });
      return Array.isArray(response.data) ? response.data : [];
    } catch (error) {
      console.error('Error fetching feed:', error);
      return [];
    }
  },
  
  // Search endpoint
  searchSkills: async (userId, query) => {
    try {
      const response = await axios.get(`/search/${userId}`, {
        params: { query },
        timeout: 10000 // 10 second timeout
      });
      
      console.log('Raw API response:', response);
      return {
        success: true,
        data: Array.isArray(response.data) ? response.data : []
      };
    } catch (error) {
      console.error('API Error:', {
        request: error.config,
        response: error.response,
        message: error.message
      });
      
      return {
        success: false,
        error: error.response?.data || error.message
      };
    }
  },

  // src/services/api.js (add these methods)
updateUser: (userId, userData) => {
  return axios.put(`${API_URL}/users/${userId}`, userData);
},

uploadProfileImage: (userId, imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('user_id', userId);
  
  return axios.post(`${API_URL}/users/${userId}/profile-image`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
},

};

export default api;
