// src/pages/TransactionsPage.jsx
import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  Box, 
  Tabs, 
  Tab, 
  CircularProgress 
} from '@mui/material';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import api from '../services/api';
import TransactionItem from '../components/transactions/TransactionItem';

const TransactionsPage = () => {
  const { currentUser, refreshUser } = useAuth();
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);
  
  const fetchTransactions = async () => {
    if (currentUser) {
      setLoading(true);
      try {
        const response = await api.getUserTransactions(currentUser.user_id);
        setTransactions(response.data);
      } catch (error) {
        console.error('Error fetching transactions:', error);
      } finally {
        setLoading(false);
      }
    }
  };
  
  useEffect(() => {
    fetchTransactions();
  }, [currentUser]);
  
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };
  
  const handleStatusUpdate = () => {
    fetchTransactions();
    refreshUser();
  };
  
  const filteredTransactions = transactions.filter(transaction => {
    if (tabValue === 0) return true; // All
    if (tabValue === 1) return transaction.status === 'pending';
    if (tabValue === 2) return transaction.status === 'ongoing';
    if (tabValue === 3) return transaction.status === 'completed';
    return false;
  });
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };
  
  return (
    <Container maxWidth="md">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          My Skill Exchanges
        </Typography>
        
        <Tabs
          value={tabValue}
          onChange={handleTabChange}
          indicatorColor="primary"
          textColor="primary"
          variant="fullWidth"
          sx={{ mb: 4 }}
        >
          <Tab label="All" />
          <Tab label="Pending" />
          <Tab label="Ongoing" />
          <Tab label="Completed" />
        </Tabs>
        
        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {filteredTransactions.length > 0 ? (
              <motion.div
                variants={containerVariants}
                initial="hidden"
                animate="visible"
              >
                {filteredTransactions.map((transaction) => (
                  <TransactionItem 
                    key={transaction.transaction_id}
                    transaction={transaction}
                    onStatusUpdate={handleStatusUpdate}
                  />
                ))}
              </motion.div>
            ) : (
              <Box sx={{ textAlign: 'center', my: 4 }}>
                <Typography variant="body1" color="text.secondary">
                  No transactions found in this category.
                </Typography>
              </Box>
            )}
          </>
        )}
      </Box>
    </Container>
  );
};

export default TransactionsPage;
