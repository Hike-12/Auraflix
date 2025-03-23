import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { User, Image, Heart, TrendingUp, Award, Calendar, MessageCircle, BarChart2 } from "lucide-react";
import { useNavigate } from "react-router-dom";
import InfluencerComparison from "./InfluencerComparision"; // Make sure path is correct

const InfluencerSuggestions = () => {
  const [filteredUsers, setFilteredUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showComparisonSidebar, setShowComparisonSidebar] = useState(false);
  const [selectedForComparison, setSelectedForComparison] = useState([]);
  const [showComparisonView, setShowComparisonView] = useState(false); // New state to control comparison view
  const navigate = useNavigate();

  const handleNavigateToAnalysis = (influencerData) => {
    const username = influencerData.channel_info;
    navigate(`/influencers/${username}`, { state: { influencer: influencerData } });
  };

  const handleAddToComparison = (influencer) => {
    if (selectedForComparison.length >= 2) {
      // Replace the second one
      setSelectedForComparison([selectedForComparison[0], influencer]);
    } else if (selectedForComparison.find(item => item.channel_info === influencer.channel_info)) {
      // Already selected, remove it
      setSelectedForComparison(selectedForComparison.filter(item => item.channel_info !== influencer.channel_info));
    } else {
      // Add to selection
      setSelectedForComparison([...selectedForComparison, influencer]);
    }
  };

  const handleComparisonClick = () => {
    setShowComparisonView(true); // Show comparison component
    setShowComparisonSidebar(false); // Hide sidebar when showing comparison
  };

  // Function to go back to suggestions view
  const handleBackToSuggestions = () => {
    setShowComparisonView(false);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const stored = localStorage.getItem("suggestedInfluencers");
        if (!stored) {
          setError("No suggestions found in localStorage.");
          setLoading(false);
          return;
        }

        const parsedSuggestions = JSON.parse(stored);
        const usernames = parsedSuggestions.flatMap((sugg) =>
          sugg.username.split("\n").map((uname) => uname.trim().toLowerCase())
        );

        const response = await fetch("http://127.0.0.1:8000/users");
        if (!response.ok) {
          throw new Error("Failed to fetch user data from server.");
        }

        const data = await response.json();
        const allUsers = data.users;

        const filtered = allUsers.filter((user) =>
          usernames.includes(user.channel_info.toLowerCase())
        );

        setFilteredUsers(filtered);
      } catch (err) {
        console.error(err);
        setError(err.message || "Something went wrong.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const formatDisplayNumber = (num) => {
    if (!num) return "0";
    
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    } else {
      return num.toString();
    }
  };

  const parseFormattedNumber = (formatted) => {
    if (!formatted || typeof formatted !== 'string') return 0;
    
    if (formatted.endsWith('k') || formatted.endsWith('K')) {
      return parseFloat(formatted) * 1000;
    } else if (formatted.endsWith('m') || formatted.endsWith('M')) {
      return parseFloat(formatted) * 1000000;
    } else if (formatted.endsWith('b') || formatted.endsWith('B')) {
      return parseFloat(formatted) * 1000000000;
    }
    
    return parseFloat(formatted) || 0;
  };

  const calculateEngagementRate = (likes, followers) => {
    const likesNum = typeof likes === 'number' ? likes : parseFormattedNumber(likes);
    const followersNum = typeof followers === 'number' ? followers : parseFormattedNumber(followers);
    
    if (!followersNum) return "0";
    return ((likesNum / followersNum) * 100).toFixed(2);
  };

  // Render influencer comparison if showComparisonView is true
  if (showComparisonView && selectedForComparison.length === 2) {
    return (
      <InfluencerComparison 
        initialInfluencers={selectedForComparison} 
        onGoBack={handleBackToSuggestions} 
      />
    );
  }

  // Otherwise render the suggestions view
  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
      <div className="relative flex-1 flex overflow-hidden bg-black/[0.96] antialiased">
        <div
          className="pointer-events-none absolute inset-0 select-none"
          style={{
            backgroundImage:
              "linear-gradient(to right, #171717 1px, transparent 1px), linear-gradient(to bottom, #171717 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="relative z-10 w-full max-w-5xl mx-auto px-6 py-20">
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="bg-gradient-to-b from-neutral-50 to-neutral-400 bg-clip-text text-center text-4xl md:text-7xl font-bold text-transparent mb-12"
          >
            Suggested Influencers
          </motion.h1>

          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-2xl mx-auto p-4 mb-6 bg-blue-500/20 border border-blue-500/30 text-blue-300 rounded-md text-center"
            >
              <p>Loading suggestions...</p>
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-2xl mx-auto p-4 mb-6 bg-red-500/20 border border-red-500/30 text-red-300 rounded-md"
            >
              <p>{error}</p>
            </motion.div>
          )}

          {!loading && filteredUsers.length === 0 && !error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="max-w-2xl mx-auto p-4 mb-6 bg-yellow-500/20 border border-yellow-500/30 text-yellow-300 rounded-md text-center"
            >
              <p>No matching influencers found in platform data.</p>
            </motion.div>
          )}

          {!loading && filteredUsers.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {filteredUsers.map((user, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className="border border-white/20 rounded-2xl overflow-hidden bg-black/30 backdrop-blur-sm"
                  whileHover={{ scale: 1.02, transition: { duration: 0.2 } }}
                >
                  {/* Header Section */}
                  <div className="bg-gradient-to-r from-purple-900/70 to-indigo-900/70 p-4 border-b border-white/10">
                    <div className="flex items-center gap-4">
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.3 + index * 0.1, type: "spring" }}
                        className="w-16 h-16 bg-white/10 backdrop-blur-md rounded-full flex items-center justify-center text-white font-bold text-2xl"
                      >
                        {user.channel_info.charAt(0).toUpperCase()}
                      </motion.div>
                      <motion.div
                        initial={{ x: -20, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        transition={{ delay: 0.4 + index * 0.1 }}
                      >
                        <h3 className="text-xl font-bold">@{user.channel_info}</h3>
                        <div className="flex items-center mt-1 text-white/70">
                          <div className="flex items-center gap-1">
                            <span className="text-sm">📍</span>
                            <span>{user.country || "N/A"}</span>
                          </div>
                          <span className="mx-2">•</span>
                          <div className="flex items-center gap-1">
                            <span className="text-sm">✨</span>
                            <span>Rank #{user.rank}</span>
                          </div>
                        </div>
                      </motion.div>
                    </div>
                  </div>

                  {/* Stats Cards */}
                  <div className="p-4">
                    <motion.h4
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      className="text-lg font-semibold text-white/90 mb-3"
                    >
                      Performance Metrics
                    </motion.h4>

                    <div className="grid grid-cols-2 gap-3 mb-4">
                      {[
                        {
                          title: "Followers",
                          value: formatDisplayNumber(user.followers),
                          icon: <User size={16} className="text-blue-400" />,
                          gradient: "from-blue-900/30 to-blue-800/30",
                        },
                        {
                          title: "Posts",
                          value: formatDisplayNumber(user.posts),
                          icon: <Image size={16} className="text-green-400" />,
                          gradient: "from-green-900/30 to-green-800/30",
                        },
                        {
                          title: "Avg. Likes",
                          value: formatDisplayNumber(user.avg_likes),
                          icon: <Heart size={16} className="text-pink-400" />,
                          gradient: "from-pink-900/30 to-pink-800/30",
                        },
                        {
                          title: "Engagement",
                          value: `${calculateEngagementRate(
                            user.avg_likes,
                            user.followers
                          )}%`,
                          icon: <TrendingUp size={16} className="text-amber-400" />,
                          gradient: "from-amber-900/30 to-amber-800/30",
                        },
                      ].map((stat, statIndex) => (
                        <motion.div
                          key={statIndex}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: 0.6 + index * 0.1 + statIndex * 0.1 }}
                          whileHover={{ scale: 1.05 }}
                          className={`bg-gradient-to-br ${stat.gradient} p-3 rounded-xl border border-white/10 backdrop-blur-sm`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-white/70 text-sm font-medium">
                              {stat.title}
                            </p>
                            {stat.icon}
                          </div>
                          <p className="text-lg font-bold text-white">
                            {stat.value}
                          </p>
                        </motion.div>
                      ))}
                    </div>

                    {/* Additional Stats */}
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1 + index * 0.1 }}
                      className="bg-white/5 border border-white/10 p-4 rounded-xl mb-4"
                    >
                      <h4 className="text-md font-semibold text-white/90 mb-3">
                        Score Metrics
                      </h4>
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                        {[
                          {
                            label: "InfluenceIQ",
                            value: user.influenceiq_score,
                            icon: <Award size={14} className="text-purple-400" />
                          },
                          {
                            label: "Influence Score",
                            value: user.influence_score,
                            icon: <TrendingUp size={14} className="text-blue-400" />
                          },
                          {
                            label: "Credibility",
                            value: user.credibility_score,
                            icon: <BarChart2 size={14} className="text-green-400" />
                          },
                          {
                            label: "Engagement Quality",
                            value: user.engagement_quality_score,
                            icon: <MessageCircle size={14} className="text-pink-400" />
                          },
                          {
                            label: "Longevity",
                            value: user.longevity_score,
                            icon: <Calendar size={14} className="text-amber-400" />
                          },
                          {
                            label: "Total Likes",
                            value: formatDisplayNumber(user.total_likes),
                            icon: <Heart size={14} className="text-red-400" />
                          }
                        ].map((item, itemIndex) => (
                          <motion.div 
                            key={itemIndex}
                            className="transform transition-all duration-500 hover:translate-x-1 flex items-center gap-2"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 1.1 + index * 0.1 + itemIndex * 0.05 }}
                          >
                            {item.icon}
                            <div>
                              <p className="text-xs text-white/50">{item.label}</p>
                              <p className="text-sm font-bold text-white">{item.value}</p>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>

                    <motion.a
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1.2 + index * 0.1 }}
                      href={`https://www.instagram.com/${user.channel_info}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block w-full text-center bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 py-2 rounded-lg text-white font-medium transition-all"
                      whileHover={{ scale: 1.03 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      View Profile
                    </motion.a>
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 1.2 + index * 0.1 }}
                      className="flex justify-center mt-3"
                    >
                      <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.95 }}
                        className="bg-white text-black px-4 py-2 rounded-md hover:bg-opacity-90 transition-all font-medium flex items-center gap-2 text-sm"
                        onClick={() => handleNavigateToAnalysis(user)} 
                      >
                        View Detailed Analytics
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="m9 18 6-6-6-6" />
                        </svg>
                      </motion.button>
                    </motion.div>

                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleAddToComparison(user);
                      }}
                      className={`mt-2 w-full bg-gray-800 text-white px-4 py-2 rounded-md hover:bg-gray-700 transition-all font-medium flex items-center justify-center gap-2 text-sm ${
                        selectedForComparison.some(inf => inf.channel_info === user.channel_info) 
                          ? 'bg-purple-800 hover:bg-purple-700' 
                          : ''
                      }`}
                    >
                      {selectedForComparison.some(inf => inf.channel_info === user.channel_info) 
                        ? 'Remove from Comparison' 
                        : 'Add to Comparison'}
                    </motion.button>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
      <motion.div 
        className={`fixed right-0 top-0 h-full w-72 bg-black/90 border-l border-white/20 p-4 transform transition-all duration-300 z-50 ${showComparisonSidebar ? 'translate-x-0' : 'translate-x-full'}`}
        initial={false}
      >
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-white">Compare Influencers</h3>
          <button 
            onClick={() => setShowComparisonSidebar(false)}
            className="text-white/70 hover:text-white"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18"></path>
              <path d="M6 6l12 12"></path>
            </svg>
          </button>
        </div>
        
        <div className="mb-4">
          <p className="text-white/70 text-sm mb-2">Selected for comparison ({selectedForComparison.length}/2):</p>
          <div className="space-y-2">
            {selectedForComparison.map((inf, idx) => (
              <div key={idx} className="bg-white/10 p-2 rounded flex justify-between items-center">
                <span>@{inf.channel_info}</span>
                <button 
                  onClick={() => handleAddToComparison(inf)}
                  className="text-red-400 hover:text-red-300"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 6L6 18"></path>
                    <path d="M6 6l12 12"></path>
                  </svg>
                </button>
              </div>
            ))}
            
            {selectedForComparison.length < 2 && (
              <div className="bg-white/5 p-2 rounded border border-dashed border-white/20 text-center text-white/50 text-sm">
                Select {2 - selectedForComparison.length} more influencer{selectedForComparison.length === 0 ? 's' : ''}
              </div>
            )}
          </div>
        </div>
        
        <button
          onClick={handleComparisonClick}
          disabled={selectedForComparison.length < 2}
          className={`w-full py-2 rounded-md text-white font-medium transition-all ${
            selectedForComparison.length === 2 
              ? 'bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700' 
              : 'bg-gray-700 cursor-not-allowed opacity-50'
          }`}
        >
          Compare Influencers
        </button>
      </motion.div>

      {/* Comparison Toggle Button */}
      <motion.button
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        onClick={() => setShowComparisonSidebar(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-blue-600 p-3 rounded-full shadow-lg text-white hover:from-purple-700 hover:to-blue-700 transition-all z-40"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="21 8 21 21 3 21 3 8"></polyline>
          <rect x="1" y="3" width="22" height="5"></rect>
          <line x1="10" y1="12" x2="14" y2="12"></line>
        </svg>
      </motion.button>
    </div>
  );
};

export default InfluencerSuggestions;