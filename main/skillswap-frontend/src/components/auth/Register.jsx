// src/components/auth/Register.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  TextField, 
  Button, 
  Typography, 
  Box, 
  Paper, 
  Stepper, 
  Step, 
  StepLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  OutlinedInput
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { SKILL_OPTIONS, GENDER_OPTIONS, LANGUAGE_OPTIONS } from '../../utils/constants';

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
  PaperProps: {
    style: {
      maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
      width: 250,
    },
  },
};

const steps = ['Account Information', 'Personal Details', 'Skills Selection'];

const Register = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    bio: '',
    gender: '',
    can_teach_list: [],
    wants_to_learn_list: [],
    can_teach_description: '',
    wants_to_learn_description: '',
  });
  const [profileImage, setProfileImage] = useState(null);
  const [error, setError] = useState('');
  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSkillChange = (event, skillType) => {
    const {
      target: { value },
    } = event;
    
    setFormData({
      ...formData,
      [skillType]: typeof value === 'string' ? value.split(',').map(Number) : value,
    });
    
    // Also update the text description
    if (skillType === 'can_teach_list') {
      const skillNames = value.map(id => SKILL_OPTIONS[id]).join(', ');
      setFormData(prev => ({ ...prev, can_teach: skillNames }));
    } else if (skillType === 'wants_to_learn_list') {
      const skillNames = value.map(id => SKILL_OPTIONS[id]).join(', ');
      setFormData(prev => ({ ...prev, wants_to_learn: skillNames }));
    }
  };

  const handleImageChange = (e) => {
    if (e.target.files[0]) {
      setProfileImage(e.target.files[0]);
    }
  };

  const handleNext = () => {
    if (activeStep === 0) {
      // Validate account information
      if (!formData.username || !formData.email || !formData.password || !formData.confirmPassword) {
        setError('Please fill in all fields');
        return;
      }
      
      if (formData.password !== formData.confirmPassword) {
        setError('Passwords do not match');
        return;
      }
    }
    
    setError('');
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    if (formData.can_teach_list.length === 0 || formData.wants_to_learn_list.length === 0) {
      setError('Please select at least one skill to teach and one skill to learn');
      return;
    }
    
    try {
      // Remove confirmPassword before sending to API
      const { confirmPassword, ...userData } = formData;
      
      // Create FormData object for multipart/form-data
      const formDataObj = new FormData();
      
      // Convert userData object to JSON string and append to formData
      formDataObj.append('user_data', JSON.stringify(userData));
      
      // Add profile image if provided
      if (profileImage) {
        formDataObj.append('file', profileImage);
      }
      
      console.log('Submitting registration data:', userData);
      
      // Use the correct ngrok URL from your FastAPI server
      const response = await fetch('http://localhost:8000/register', {
        method: 'POST',
        body: formDataObj,
        // Don't set Content-Type header - the browser will set it automatically for FormData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        throw new Error(`Registration failed: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Registration successful:', data);
      
      // Navigate to login page on success
      navigate('/login');
    } catch (err) {
      console.error('Registration error:', err);
      setError(err.message || 'Registration failed');
    }
  };
  
  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              label="Username"
              name="username"
              autoComplete="username"
              autoFocus
              value={formData.username}
              onChange={handleChange}
            />
            
            <TextField
              margin="normal"
              required
              fullWidth
              id="email"
              label="Email Address"
              name="email"
              autoComplete="email"
              value={formData.email}
              onChange={handleChange}
            />
            
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="new-password"
              value={formData.password}
              onChange={handleChange}
            />
            
            <TextField
              margin="normal"
              required
              fullWidth
              name="confirmPassword"
              label="Confirm Password"
              type="password"
              id="confirmPassword"
              value={formData.confirmPassword}
              onChange={handleChange}
            />
          </Box>
        );
      case 1:
        return (
          <Box>
            <TextField
              margin="normal"
              fullWidth
              id="bio"
              label="Bio"
              name="bio"
              multiline
              rows={4}
              value={formData.bio}
              onChange={handleChange}
            />
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="gender-label">Gender</InputLabel>
              <FormControl fullWidth margin="normal">
  <InputLabel id="gender-label"></InputLabel>
  <Select
    labelId="gender-label"
    id="gender"
    name="gender"
    value={formData.gender}
    onChange={handleChange}
    label="Gender"
  >
    {Object.entries(GENDER_OPTIONS).map(([value, label]) => (
      <MenuItem key={value} value={value}>
        {label}
      </MenuItem>
    ))}
  </Select>
</FormControl>



            </FormControl>
            
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Profile Image
              </Typography>
              <input
                accept="image/*"
                id="profile-image"
                type="file"
                onChange={handleImageChange}
              />
            </Box>
          </Box>
        );
      case 2:
        return (
          <Box>
            <FormControl fullWidth margin="normal">
              <InputLabel id="skills-to-teach-label">Skills I Can Teach</InputLabel>
              <Select
                labelId="skills-to-teach-label"
                id="can_teach_list"
                name="can_teach_list"
                multiple
                value={formData.can_teach_list || []}
                onChange={(e) => handleSkillChange(e, 'can_teach_list')}
                input={<OutlinedInput id="select-multiple-chip" label="Skills I Can Teach" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={SKILL_OPTIONS[value]} />
                    ))}
                  </Box>
                )}
                MenuProps={MenuProps}
              >
                {Object.entries(SKILL_OPTIONS).map(([value, label]) => (
                  <MenuItem key={value} value={Number(value)}>
                    {label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
      
            {/* Add description field for skills to teach */}
            <TextField
              margin="normal"
              fullWidth
              id="can_teach_description"
              label="Describe your teaching skills further"
              name="can_teach_description"
              multiline
              rows={3}
              value={formData.can_teach_description}
              onChange={handleChange}
              sx={{ mt: 2 }}
              placeholder="Explain your experience with these skills, your teaching style, etc."
            />
      
            <FormControl fullWidth margin="normal">
              <InputLabel id="skills-to-learn-label">Skills I Want to Learn</InputLabel>
              <Select
                labelId="skills-to-learn-label"
                id="wants_to_learn_list"
                name="wants_to_learn_list"
                multiple
                value={formData.wants_to_learn_list || []}
                onChange={(e) => handleSkillChange(e, 'wants_to_learn_list')}
                input={<OutlinedInput id="select-multiple-chip" label="Skills I Want to Learn" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={SKILL_OPTIONS[value]} />
                    ))}
                  </Box>
                )}
                MenuProps={MenuProps}
              >
                {Object.entries(SKILL_OPTIONS).map(([value, label]) => (
                  <MenuItem key={value} value={Number(value)}>
                    {label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
      
            {/* Add description field for skills to learn */}
            <TextField
              margin="normal"
              fullWidth
              id="wants_to_learn_description"
              label="Describe what you want to learn"
              name="wants_to_learn_description"
              multiline
              rows={3}
              value={formData.wants_to_learn_description}
              onChange={handleChange}
              sx={{ mt: 2 }}
              placeholder="Explain what specifically you want to learn, your goals, etc."
            />
            <FormControl fullWidth margin="normal">
  <InputLabel id="languages-label">Languages You Prefer</InputLabel>
  <Select
    labelId="languages-label"
    id="languages"
    name="languages"
    multiple
    value={formData.languages || []}
    onChange={(e) => {
      e.target.value = e.target.value.map(Number); // 💥 convert to int
      handleChange(e);
    }}
    input={<OutlinedInput id="select-multiple-chip" label="Languages You Prefer" />}
    renderValue={(selected) => (
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
        {selected.map((value) => (
          <Chip key={value} label={LANGUAGE_OPTIONS[value]} />
        ))}
      </Box>
    )}
    MenuProps={MenuProps}
  >
    {Object.entries(LANGUAGE_OPTIONS).map(([value, label]) => (
      <MenuItem key={value} value={Number(value)}>
        {label}
      </MenuItem>
    ))}
  </Select>
</FormControl>

          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Paper elevation={3} sx={{ p: 4, maxWidth: 600, mx: 'auto', mt: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Join SkillSwap
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        {error && (
          <Typography color="error" align="center" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}
        
        <Box component="form" onSubmit={handleSubmit} noValidate>
          {getStepContent(activeStep)}
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
            >
              Back
            </Button>
            
            {activeStep === steps.length - 1 ? (
              <Button
                variant="contained"
                onClick={handleSubmit}
              >
                Register
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
              >
                Next
              </Button>
            )}
          </Box>
        </Box>
        
        <Typography align="center" sx={{ mt: 3 }}>
          Already have an account?{' '}
          <Button variant="text" onClick={() => navigate('/login')}>
            Login
          </Button>
        </Typography>
      </Paper>
    </motion.div>
  );
};

export default Register;
