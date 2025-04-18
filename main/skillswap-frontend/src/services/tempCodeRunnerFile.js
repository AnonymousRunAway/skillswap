  searchSkills: (userId, query) => {
    return axios.get(`${API_URL}/search/${userId}?query=${encodeURIComponent(query)}`);
  },