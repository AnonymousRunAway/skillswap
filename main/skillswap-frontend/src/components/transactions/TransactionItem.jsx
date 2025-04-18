// src/components/transactions/TransactionItem.jsx
import React, { useState } from 'react';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  CardActions, 
  Typography, 
  Button, 
  Box, 
  Chip, 
  Rating,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import { motion } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';

const TransactionItem = ({ transaction, onStatusUpdate }) => {
  const { currentUser } = useAuth();
  const [open, setOpen] = useState(false);
  const [rating, setRating] = useState(5);
  
  const isTeacher = currentUser.user_id === transaction.teacher_id;
  const isLearner = currentUser.user_id === transaction.learner_id;
  
  const canConfirm = (isTeacher && !transaction.teacher_confirmed) || 
                     (isLearner && !transaction.learner_confirmed);
  
  const handleOpenDialog = () => {
    setOpen(true);
  };
  
  const handleCloseDialog = () => {
    setOpen(false);
  };
  
  const handleConfirm = async () => {
    try {
      await api.confirmTransaction(
        transaction.transaction_id,
        currentUser.user_id,
        rating
      );
      
      if (onStatusUpdate) {
        onStatusUpdate();
      }
      
      handleCloseDialog();
    } catch (error) {
      console.error('Error confirming transaction:', error);
    }
  };
  
  const getStatusColor = (status) => {
    switch (status) {
      case 'pending':
        return '#ff9800';
      case 'ongoing':
        return '#2196f3';
      case 'completed':
        return '#4caf50';
      case 'cancelled':
        return '#f44336';
      default:
        return '#757575';
    }
  };
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ mb: 3 }}>
        <CardHeader
          title={`Skill Exchange: ${transaction.skill_taught} ⟷ ${transaction.skill_learned}`}
          subheader={
            <Chip 
              label={transaction.status.toUpperCase()}
              size="small"
              sx={{ 
                bgcolor: getStatusColor(transaction.status),
                color: 'white',
                mt: 1
              }}
            />
          }
        />
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                You will {isTeacher ? 'teach' : 'learn'}:
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {isTeacher ? transaction.skill_taught : transaction.skill_learned}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="body2" color="text.secondary">
                You will {isTeacher ? 'learn' : 'teach'}:
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {isTeacher ? transaction.skill_learned : transaction.skill_taught}
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                Teacher confirmed:
              </Typography>
              <Typography variant="body1">
                {transaction.teacher_confirmed ? 'Yes' : 'No'}
              </Typography>
            </Box>
            
            <Box>
              <Typography variant="body2" color="text.secondary">
                Learner confirmed:
              </Typography>
              <Typography variant="body1">
                {transaction.learner_confirmed ? 'Yes' : 'No'}
              </Typography>
            </Box>
          </Box>
        </CardContent>
        
        {canConfirm && (
          <CardActions>
            <Button 
              variant="contained" 
              fullWidth
              onClick={handleOpenDialog}
            >
              Confirm Completion
            </Button>
          </CardActions>
        )}
      </Card>
      
      <Dialog open={open} onClose={handleCloseDialog}>
        <DialogTitle>Confirm Skill Exchange</DialogTitle>
        <DialogContent>
          <Typography variant="body1" gutterBottom>
            Please rate your experience:
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', my: 2 }}>
            <Rating
              value={rating}
              onChange={(event, newValue) => {
                setRating(newValue);
              }}
              precision={0.5}
              size="large"
            />
            <Typography variant="body1" sx={{ ml: 2 }}>
              {rating}/5
            </Typography>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button onClick={handleConfirm} variant="contained">
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </motion.div>
  );
};

export default TransactionItem;
