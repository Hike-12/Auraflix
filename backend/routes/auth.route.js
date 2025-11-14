// routes/auth.js
const express = require("express");
const router = express.Router();
const BusinessUser = require("../models/user.model");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const axios = require("axios");

// Register a new business
router.post("/register", async (req, res) => {
  try {
    const {
      businessName,
      email,
      password,
      businessCategory,
      description,
      contact,
      website,
      socialMedia,
    } = req.body;

    // Check if user already exists
    const existingUser = await BusinessUser.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "Business already registered" });
    }

    let suggestions = [];
    
    // Get suggestions from FastAPI before creating the user
    try {
      const apiResponse = await axios.post(
        "https://influenceiq-python.onrender.com/receive-business",
        {
          businessName,
          email,
          businessCategory,
          description,
          contact,
          website,
          socialMedia,
        }
      );

      console.log("Data sent to FastAPI:", apiResponse.data);
      suggestions = apiResponse.data.suggested_influencers || [];
    } catch (apiError) {
      console.error(
        "Failed to get suggestions from FastAPI:",
        apiError.message
      );
      // Continue with registration even if API fails
    }

    // Create new business user with suggestions included
    const businessUser = new BusinessUser({
      businessName,
      email,
      password,
      businessCategory,
      description,
      contact,
      website,
      socialMedia,
      suggestedInfluencers: suggestions
    });

    await businessUser.save();

    // Generate JWT token
    const token = jwt.sign({ id: businessUser._id }, process.env.JWT_SECRET, {
      expiresIn: "7d",
    });

    // Send response with suggestions
    res.status(201).json({
      token,
      businessUser: businessUser.toObject(),
      suggestions: suggestions,
    });
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Login for business
router.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    // Check if user exists
    const businessUser = await BusinessUser.findOne({ email });
    if (!businessUser) {
      return res.status(400).json({ message: "Invalid credentials" });
    }

    // Check password
    const isMatch = await bcrypt.compare(password, businessUser.password);
    if (!isMatch) {
      return res.status(400).json({ message: "Invalid credentials" });
    }

    // Generate JWT token
    const token = jwt.sign({ id: businessUser._id }, process.env.JWT_SECRET, {
      expiresIn: "1h",
    });

    await axios
      .post("https://influenceiq-python.onrender.com/receive-business", {
        businessName: businessUser.businessName,
        email: businessUser.email,
        businessCategory: businessUser.businessCategory,
        description: businessUser.description,
        contact: businessUser.contact,
        website: businessUser.website,
        socialMedia: businessUser.socialMedia,
      })
      .then(async (response) => {
        console.log("FastAPI Suggestions on Login:", response.data);
        const suggestions = response.data.suggested_influencers || [];

        // Update user document with suggestions
        await BusinessUser.findByIdAndUpdate(businessUser._id, {
          suggestedInfluencers: suggestions,
        });

        // Include updated suggestions in response
        const updatedUser = {
          ...businessUser.toObject(),
          suggestedInfluencers: suggestions,
        };
        console.log("Updated user with suggestions:", updatedUser);
        res.status(200).json({
          token,
          businessUser: updatedUser,
          suggestions: suggestions,
        });
      })
      .catch((err) => {
        console.error(
          "Failed to get suggestions from FastAPI on login:",
          err.message
        );
        // Still return login success even if FastAPI call fails
        res.status(200).json({ token, businessUser });
      });
  } catch (error) {
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

module.exports = router;
