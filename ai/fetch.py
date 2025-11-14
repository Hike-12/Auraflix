from fastapi import FastAPI, HTTPException, Query
import json
import logging
import math
import random
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.api_core.exceptions import GoogleAPICallError, NotFound
from dotenv import load_dotenv
import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from io import StringIO
from fastapi.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JSON_FILE_PATH = 'output_file.json'

try:
    with open(JSON_FILE_PATH, 'r') as json_file:
        json_data = json.load(json_file)
    logger.info("JSON data loaded successfully")
except FileNotFoundError:
    raise Exception(f"JSON file not found at {JSON_FILE_PATH}")
except json.JSONDecodeError:
    raise Exception(f"Invalid JSON format in {JSON_FILE_PATH}")

# ------------------------------------- Romeiro's code started here -----------------------------------------

class InfluencerRequest(BaseModel):
    influencer: Dict[str, Any]

@app.post("/generate-summary")
async def generate_summary(request: InfluencerRequest):
    try:
        influencer = request.influencer

        prompt = f"""
        Create a professional, concise summary of this social media influencer:
        
        Username: {influencer.get('channel_info')}
        Followers: {influencer.get('followers')}
        Posts: {influencer.get('posts')}
        Average Likes: {influencer.get('avg_likes')}
        Engagement Rate: {influencer.get('avg_engagement')}
        Country: {influencer.get('country', 'Unknown')}
        Influence Score: {influencer.get('influence_score')}
        
        Include insights about their engagement quality (score: {influencer.get('engagement_quality_score', 'N/A')}),
        content longevity (score: {influencer.get('longevity_score', 'N/A')}),
        and overall influence potential. Keep the summary to 3-4 sentences and focus on what makes this
        influencer's profile notable based on these metrics.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        summary = response.text
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
    

# Add this function to help with username extraction
def get_username(channel_info):
    """Extract username from channel info"""
    if not channel_info:
        return "unknown"
    return channel_info[1:] if channel_info.startswith("@") else channel_info

# Add this new endpoint
@app.get("/trends/{username}")
async def get_influencer_trends(username: str):
    """
    Generate realistic trend data for a specific influencer based on their metrics.
    """
    logger.info(f"Generating trend data for influencer: {username}")
    
    # Find the influencer in our data
    influencer = next((item for item in json_data if get_username(item.get("channel_info", "")) == username), None)
    
    if not influencer:
        logger.warning(f"Influencer '{username}' not found, generating generic data")
        # If we can't find the influencer, generate some generic data
        return generate_generic_trend_data()
    
    # Generate last 12 months of data based on influencer metrics
    data = []
    current_date = datetime.now()
    
    # Parse metrics for data generation
    metrics = parse_influencer_metrics(influencer)
    
    # Generate timeline data points
    for i in range(-11, 1):  # Last 12 months
        month_date = current_date + timedelta(days=30 * i)
        month_name = month_date.strftime("%b")
        month_year = month_date.strftime("%b %Y")
        
        # Calculate metrics for this month with realistic trends
        month_data = calculate_month_metrics(metrics, i, influencer)
        month_data["month"] = month_name
        month_data["period"] = month_year
        
        data.append(month_data)
    
    return data

def parse_influencer_metrics(influencer):
    """Extract and normalize metrics from influencer data"""
    metrics = {}
    
    # Convert string metrics like "1.2M" to actual numbers
    try:
        # Handle followers
        followers_str = influencer.get('followers', '10K')
        if isinstance(followers_str, str):
            if 'k' in followers_str.lower() or 'K' in followers_str:
                metrics['followers'] = float(followers_str.lower().replace('k', '').replace('K', '').strip()) * 1000
            elif 'm' in followers_str.lower() or 'M' in followers_str:
                metrics['followers'] = float(followers_str.lower().replace('m', '').replace('M', '').strip()) * 1000000
            elif 'b' in followers_str.lower() or 'B' in followers_str:
                metrics['followers'] = float(followers_str.lower().replace('b', '').replace('B', '').strip()) * 1000000000
            else:
                metrics['followers'] = float(followers_str.replace(',', '').strip())
        else:
            metrics['followers'] = float(followers_str)
    except (ValueError, TypeError):
        metrics['followers'] = 10000  # Default
    
    # Handle engagement rate
    try:
        engagement_str = influencer.get('avg_engagement', '1%')
        if isinstance(engagement_str, str) and '%' in engagement_str:
            metrics['engagement'] = float(engagement_str.replace('%', '').strip())
        else:
            metrics['engagement'] = float(engagement_str) * 100 if float(engagement_str) < 1 else float(engagement_str)
    except (ValueError, TypeError):
        metrics['engagement'] = 1.0  # Default
    
    # Get other metrics
    metrics['influence_score'] = float(influencer.get('influence_score', 50))
    metrics['engagement_quality'] = float(influencer.get('engagement_quality_score', 0.5))
    metrics['longevity'] = float(influencer.get('longevity_score', 5))
    
    return metrics

def calculate_month_metrics(base_metrics, month_offset, influencer):
    """Calculate realistic metrics for a specific month"""
    # Base values
    base_followers = base_metrics['followers']
    base_engagement = base_metrics['engagement']
    influence_score = base_metrics['influence_score']
    
    # Calculate growth rate based on influence score (higher score = faster growth)
    # Convert to a monthly growth percentage between 0.5% and 5%
    monthly_growth_rate = 1 + ((influence_score / 100) * 0.045 + 0.005)
    
    # Calculate followers with compounding growth, more recent months have higher growth
    # Apply some randomness for realism
    growth_randomness = random.uniform(0.97, 1.03)  # ±3% random variation
    followers = base_followers * (monthly_growth_rate ** (month_offset + 12)) * growth_randomness
    
    # Add seasonality effect to engagement (higher in certain months)
    # Season effect follows a sine wave with peak in summer months
    month_in_year = (datetime.now().month + month_offset) % 12
    seasonal_factor = math.sin(month_in_year / 12 * 2 * math.pi) * 0.15  # ±15% seasonal variation
    
    # For engagement, consider the influencer quality score
    engagement_quality = base_metrics['engagement_quality']
    
    # Higher quality engagement tends to be more consistent
    quality_factor = 1 - (engagement_quality / 2)  # Lower value = more consistent
    random_factor = random.uniform(-0.1, 0.1) * quality_factor
    
    # Calculate engagement with seasonality and randomness
    engagement = max(0.1, base_engagement * (1 + seasonal_factor + random_factor))
    
    # Calculate likes based on followers and engagement rate
    likes = int(followers * engagement / 100)
    
    # Final data point
    return {
        "followers": int(followers),
        "engagement": round(engagement, 2),
        "likes": likes,
        "quality_score": round(base_metrics['engagement_quality'] * (1 + random_factor/5), 3)
    }

def generate_generic_trend_data():
    """Generate generic trend data when influencer is not found"""
    data = []
    current_date = datetime.now()
    
    base_followers = 100000
    monthly_growth = 1.03  # 3% monthly growth
    base_engagement = 2.0   # 2% engagement rate
    
    for i in range(-11, 1):  # Last 12 months
        month_date = current_date + timedelta(days=30 * i)
        month_name = month_date.strftime("%b")
        month_year = month_date.strftime("%b %Y")
        
        # Simple growth model
        followers = int(base_followers * (monthly_growth ** (i + 12)))
        engagement = base_engagement + (i * 0.05)  # Slight increase in engagement over time
        likes = int(followers * engagement / 100)
        
        data.append({
            "month": month_name,
            "period": month_year,
            "followers": followers,
            "engagement": round(engagement, 2),
            "likes": likes,
            "quality_score": round(0.5 + (i * 0.02), 3)  # Gradually improving quality score
        })
    
    return data



# Collab Simulator part



class BusinessDetails(BaseModel):
    businessName: str
    businessCategory: str
    description: str

class CollabRequest(BaseModel):
    business: BusinessDetails
    influencer_username: str

class CollabSimulationResponse(BaseModel):
    match_percentage: float
    estimated_cost: float
    estimated_roi: float
    estimated_reach: int
    relevancy_score: float
    longevity_potential: float
    recommendation: str

# Create a simple in-memory cache for our vectorized data
tfidf_vectorizer = None
influencer_vectors = None
influencer_map = {}

# Add this at the beginning of your file if not already present
from typing import Dict, List

# Add this cache to store Gemini-enhanced categories
enhanced_categories_cache = {
    
}

class VectorizationManager:

    @staticmethod
    def enrich_categories():
        """Load enhanced categories from JSON file instead of calling Gemini"""
        global enhanced_categories_cache
        
        try:
            import os
            import json
            
            # Path to the enhanced categories JSON file
            categories_file = "enhanced_categories.json"
            
            if not os.path.exists(categories_file):
                logger.warning(f"Categories file '{categories_file}' not found. Using default categories.")
                return
                
            logger.info(f"Loading enhanced categories from {categories_file}...")
            
            # Load the categories from file
            with open(categories_file, 'r', encoding='utf-8') as f:
                enhanced_categories = json.load(f)
                
            # Populate the cache with loaded categories
            enhanced_categories_cache.update(enhanced_categories)
            
            logger.info(f"Loaded {len(enhanced_categories_cache)} enhanced categories")
            
            # Log some sample categories for verification
            sample_size = min(5, len(enhanced_categories_cache))
            sample_entries = list(enhanced_categories_cache.items())[:sample_size]
            logger.info(f"Sample categories: {dict(sample_entries)}")
            
        except Exception as e:
            logger.error(f"Failed to load enhanced categories: {str(e)}")
            logger.error("Using default categories instead")
        
    @staticmethod
    def initialize():
        global tfidf_vectorizer, influencer_vectors, influencer_map
        
        # First enrich categories with Gemini
        VectorizationManager.enrich_categories()
        
        # Initialize TF-IDF vectorizer with improved parameters
        tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=1,
            ngram_range=(1, 2),   # Include bigrams for better context
            max_features=5000,    # Limit features to reduce noise
            use_idf=True,         # Use IDF for better term weighting
            sublinear_tf=True     # Apply sublinear TF scaling for better results with varied text lengths
        )
        
        # Extract descriptions and create a corpus with more comprehensive data
        descriptions = []
        keywords_list = []
        
        for idx, influencer in enumerate(json_data):
            # Get all text-based data
            description = influencer.get("description", "")
            channel = influencer.get("channel_info", "")
            username = get_username(channel)
            
            # Use enhanced category if available, otherwise fallback
            if username in enhanced_categories_cache:
                category = enhanced_categories_cache[username]
                # Also update the original data
                influencer["category"] = category
            else:
                category = influencer.get("category", "")
                
            country = influencer.get("country", "")
            keywords = influencer.get("keywords", "")  # Assuming you might have keywords in your data
            posts_content = str(influencer.get("content_topics", ""))  # Any content topics if available
            
            # Store username to influencer mapping
            influencer_map[username] = idx
            
            # Weight important terms by repeating them
            # This makes category and description words more influential
            weighted_text = (
                f"{description} {description} "  # Repeat for emphasis
                f"{category} {category} {category} "  # Category is very important - repeat 3x
                f"{keywords} {keywords} "  # Keywords are important - repeat 2x
                f"{country} {posts_content} {channel}"
            ).strip()
            
            # Fallback for empty descriptions
            if not weighted_text:
                weighted_text = f"influencer content creator social media {category}"
                
            descriptions.append(weighted_text)
            
            # Save keywords separately for additional matching
            keywords_list.append(f"{category} {keywords}")
        
        try:
            # Vectorize the enriched descriptions
            if not descriptions:
                logger.warning("No influencer descriptions found, using default vectors")
                descriptions = ["default content creator"]
                
            # Fit and transform our enhanced corpus
            influencer_vectors = tfidf_vectorizer.fit_transform(descriptions)
            logger.info(f"Vectorization initialized with {len(descriptions)} influencers")
            
        except ValueError as e:
            logger.error(f"Vectorization error: {str(e)}")
            # Create a fallback vectorizer
            tfidf_vectorizer = TfidfVectorizer(stop_words=None)
            dummy_texts = [f"influencer{i} content" for i in range(len(json_data))]
            influencer_vectors = tfidf_vectorizer.fit_transform(dummy_texts)
            logger.warning("Using fallback vectorization due to empty vocabulary")

# Initialize vectorization on module import
vectorization_manager = VectorizationManager()
vectorization_manager.initialize()

@app.post("/collab-simulation")
async def simulate_collaboration(request: CollabRequest):
    """
    Simulate a collaboration between a business and an influencer.
    Calculates compatibility, costs, ROI estimates, and other metrics.
    """
    logger.info(f"Simulating collab between {request.business.businessName} and {request.influencer_username}")
    
    # Find the influencer
    influencer_username = request.influencer_username
    influencer = next((item for item in json_data if get_username(item.get("channel_info", "")) == influencer_username), None)
    
    if not influencer:
        raise HTTPException(status_code=404, detail=f"Influencer '{influencer_username}' not found")
    
    # Calculate match percentage using improved algorithm
    match_percentage = calculate_match_percentage(request.business, influencer)
    
    # Calculate cost estimate based on followers and engagement
    cost_estimate = calculate_cost_estimate(influencer)
    
    # Calculate estimated ROI with improved formula
    roi_estimate = calculate_roi_estimate(match_percentage, influencer, request.business)
    
    # Calculate estimated reach
    estimated_reach = calculate_reach(influencer)
    
    # Get relevancy score
    relevancy_score = calculate_relevancy(request.business, influencer)
    
    # Calculate longevity potential
    longevity_potential = calculate_longevity(influencer)
    
    # Generate recommendation
    recommendation = generate_recommendation(match_percentage, roi_estimate, request.business, influencer)
    
    return {
        "match_percentage": round(match_percentage, 1),
        "estimated_cost": round(cost_estimate, 2),
        "estimated_roi": round(roi_estimate, 2),
        "estimated_reach": estimated_reach,
        "relevancy_score": round(relevancy_score, 2),
        "longevity_potential": round(longevity_potential, 2),
        "recommendation": recommendation
    }

def calculate_match_percentage(business: BusinessDetails, influencer: Dict[str, Any]) -> float:
    """
    Calculate the match percentage between business and influencer using 
    multiple factors including enhanced TF-IDF, accurate category match, and engagement quality
    """
    global tfidf_vectorizer, influencer_vectors, influencer_map
    
    try:
        # Adjust component weights to emphasize category match
        tfidf_weight = 0.4      # Reduced from 0.5
        category_weight = 0.4    # Increased from 0.3
        metrics_weight = 0.2     # Keep the same
        
        # Get TF-IDF similarity
        # Create a comprehensive business description
        business_text = f"{business.businessName} {business.businessCategory} {business.description}"
        
        # Handle empty business text
        if not business_text.strip():
            tfidf_similarity = 0.5  # Default for empty text
        else:
            # Transform using existing vocabulary
            business_vector = tfidf_vectorizer.transform([business_text])
            
            # Get the influencer's vector
            influencer_username = get_username(influencer.get("channel_info", ""))
            if influencer_username in influencer_map:
                idx = influencer_map[influencer_username]
                influencer_vector = influencer_vectors[idx]
                
                # Calculate cosine similarity
                tfidf_similarity = cosine_similarity(business_vector, influencer_vector)[0][0]
            else:
                tfidf_similarity = 0.5  # Default if not found
        
        # Get enhanced category match score (using Gemini-improved categories)
        business_category = business.businessCategory.lower()
        
        # Use enhanced category if available
        if influencer_username in enhanced_categories_cache:
            influencer_category = enhanced_categories_cache[influencer_username].lower()
        else:
            influencer_category = influencer.get("category", "").lower()
        
        # More sophisticated category matching
        # Check for exact match
        if business_category == influencer_category:
            category_score = 1.0
        elif business_category in influencer_category or influencer_category in business_category:
            category_score = 0.85  # Increased from 0.8
        else:
            # Enhanced semantic matching for categories
            # Use Jaccard similarity for partial word matching
            business_words = set(business_category.lower().split())
            influencer_words = set(influencer_category.lower().split())
            
            # Add synonyms for common business categories
            business_expanded = expand_category_with_synonyms(business_category)
            influencer_expanded = expand_category_with_synonyms(influencer_category)
            
            business_words.update(business_expanded)
            influencer_words.update(influencer_expanded)
            
            if not business_words or not influencer_words:
                category_score = 0.3
            else:
                intersection = len(business_words.intersection(influencer_words))
                union = len(business_words.union(influencer_words))
                category_score = intersection / union if union > 0 else 0.3
        
        # Calculate metrics score based on influencer quality metrics
        engagement_quality = float(influencer.get("engagement_quality_score", 1.0))
        # Normalize engagement quality to 0-1 scale
        if engagement_quality > 10:
            engagement_quality = engagement_quality / 10
        elif engagement_quality > 1:
            engagement_quality = engagement_quality / 5
        
        credibility = float(influencer.get("credibility_score", 50)) / 100
        influence = float(influencer.get("influence_score", 50)) / 100
        
        # Weighted metrics score with emphasis on engagement quality
        metrics_score = (engagement_quality * 0.5 + credibility * 0.25 + influence * 0.25)
        
        # Calculate final weighted score
        final_score = (
            tfidf_similarity * tfidf_weight +
            category_score * category_weight +
            metrics_score * metrics_weight
        )
        
        # Apply more aggressive power curve to better differentiate scores
        # This ensures we get more separation between good and great matches
        scaled_score = (final_score ** 0.65) * 100
        
        return max(10, min(100, scaled_score))  # Minimum 10, maximum 100
        
    except Exception as e:
        logger.error(f"Error calculating match percentage: {str(e)}")
        return 50.0  # Default match on error

def expand_category_with_synonyms(category: str) -> List[str]:
    """Add common synonyms and related terms for better category matching"""
    category = category.lower()
    synonyms = []
    
    # Map of common category synonyms
    synonym_map = {
        "fashion": ["clothing", "apparel", "style", "outfits", "wear", "dress"],
        "beauty": ["makeup", "cosmetics", "skincare", "glamour"],
        "travel": ["tourism", "vacation", "trips", "adventure", "destination"],
        "fitness": ["workout", "gym", "exercise", "health", "training", "sports"],
        "food": ["cooking", "culinary", "recipe", "dining", "cuisine", "gastronomy"],
        "tech": ["technology", "gadgets", "electronics", "digital", "computing"],
        "gaming": ["games", "videogames", "esports", "gamer"],
        "lifestyle": ["life", "living", "daily", "routine"],
        "business": ["entrepreneur", "startup", "corporate", "company"],
        "education": ["learning", "teaching", "academic", "study", "school"],
        "music": ["songs", "artist", "audio", "musical"],
        "pets": ["animals", "dogs", "cats", "pet care"],
    }
    
    # Check each key in the synonym map
    for key, values in synonym_map.items():
        if key in category:
            synonyms.extend(values)
        # Also check if any synonyms are in the category
        for value in values:
            if value in category and key not in synonyms:
                synonyms.append(key)
    
    return synonyms
    
def calculate_cost_estimate(influencer: Dict[str, Any]) -> float:
    """Calculate estimated cost for a collaboration based on influencer metrics"""
    # Parse follower count
    followers = parse_follower_count(influencer.get("followers", "10K"))
    
    # Parse engagement rate
    engagement_str = influencer.get("avg_engagement", "1%")
    if isinstance(engagement_str, str) and '%' in engagement_str:
        engagement = float(engagement_str.replace('%', '').strip())
    else:
        engagement = float(engagement_str) * 100 if float(engagement_str) < 1 else float(engagement_str)
    
    # Use a more realistic cost model: 
    # Base formula adjusted for follower count scale with diminishing returns
    if followers < 10000:  # Micro
        base_cost = (followers / 1000) * 0.8
    elif followers < 100000:  # Small
        base_cost = 8 + ((followers - 10000) / 1000) * 0.6
    elif followers < 1000000:  # Medium
        base_cost = 62 + ((followers - 100000) / 1000) * 0.4
    else:  # Large/Mega
        base_cost = 422 + ((followers - 1000000) / 1000) * 0.2
    
    # Adjust for engagement (higher engagement = higher cost)
    # Industry average is ~1-3% engagement, so we'll use that as baseline
    engagement_multiplier = (engagement / 2.0) ** 0.8  # Diminishing returns
    
    # Account for influence score
    influence_score = float(influencer.get("influence_score", 50))
    influence_multiplier = (influence_score / 50) ** 0.7  # Diminishing returns
    
    # Account for credibility
    credibility_score = float(influencer.get("credibility_score", 50))
    credibility_multiplier = (credibility_score / 50) ** 0.5  # Smaller adjustment
    
    # Ensure minimum cost
    min_cost = 50  # Minimum $50 per collab
    
    # Final cost
    calculated_cost = base_cost * engagement_multiplier * influence_multiplier * credibility_multiplier
    return max(min_cost, calculated_cost)

def calculate_roi_estimate(match_percentage: float, influencer: Dict[str, Any], business: BusinessDetails) -> float:
    """
    Estimate ROI based on match percentage, influencer metrics, and business details.
    Improved to provide more realistic ROI estimates.
    """
    # Parse metrics
    followers = parse_follower_count(influencer.get("followers", "10K"))
    
    # Get engagement rate
    engagement_str = influencer.get("avg_engagement", "1%")
    if isinstance(engagement_str, str) and '%' in engagement_str:
        engagement = float(engagement_str.replace('%', '').strip())
    else:
        engagement = float(engagement_str) * 100 if float(engagement_str) < 1 else float(engagement_str)
    
    # Get metrics relevant to ROI calculation
    credibility = float(influencer.get("credibility_score", 50))
    influence = float(influencer.get("influence_score", 50))
    
    # Get engagement quality score
    engagement_quality = float(influencer.get("engagement_quality_score", 1.0))
    # Normalize if needed
    if engagement_quality > 5:
        engagement_quality = engagement_quality / 5
    
    # Better conversion rate model:
    # Base conversion affected by match %, influencer quality, and engagement
    # Typical social media conversion rates range from 0.5% to 3%
    
    # Base conversion rate varies between 0.05% and 0.5% of engaged followers
    base_conversion = 0.0005 + (match_percentage / 100) * 0.0045
    
    # Quality multiplier based on credibility and engagement quality
    quality_multiplier = 0.5 + ((credibility / 100) * 0.25 + (engagement_quality) * 0.25)
    
    # Final conversion rate (percentage of engaged followers who convert)
    conversion_rate = base_conversion * quality_multiplier
    
    # Estimate average order value (AOV) - can be customized per industry
    aov = 50  # Default $50
    
    # Engaged audience: followers * engagement rate
    engaged_audience = followers * (engagement / 100)
    
    # Estimated conversions
    conversions = engaged_audience * conversion_rate
    
    # Estimated revenue
    estimated_revenue = conversions * aov
    
    # Estimated cost
    estimated_cost = calculate_cost_estimate(influencer)
    
    # ROI = (Revenue - Cost) / Cost * 100
    if estimated_cost > 0:
        roi = ((estimated_revenue - estimated_cost) / estimated_cost) * 100
        
        # Apply match percentage as a reality check (better matches = more accurate ROI)
        confidence_factor = 0.7 + (match_percentage / 100) * 0.3
        adjusted_roi = roi * confidence_factor
        
        return max(0, adjusted_roi)  # No negative ROI predictions
    else:
        return 0

def calculate_reach(influencer: Dict[str, Any]) -> int:
    """Calculate estimated reach based on followers and engagement with a more realistic model"""
    followers = parse_follower_count(influencer.get("followers", "10K"))
    
    # Get engagement rate
    engagement_str = influencer.get("avg_engagement", "1%")
    if isinstance(engagement_str, str) and '%' in engagement_str:
        engagement = float(engagement_str.replace('%', '').strip())
    else:
        engagement = float(engagement_str) * 100 if float(engagement_str) < 1 else float(engagement_str)
    
    # Scale reach by follower count (larger accounts typically have lower organic reach %)
    if followers < 10000:  # Micro
        base_reach_pct = 25
    elif followers < 100000:  # Small
        base_reach_pct = 20
    elif followers < 1000000:  # Medium
        base_reach_pct = 15
    else:  # Large/Mega
        base_reach_pct = 10
    
    # Adjust for engagement - higher engagement typically means better algorithm performance
    engagement_bonus = min(15, engagement * 2)
    
    # Calculate reach percentage with a cap
    reach_percentage = min(50, base_reach_pct + engagement_bonus)
    
    # Viral factor based on engagement and credibility
    credibility = float(influencer.get("credibility_score", 50))
    viral_multiplier = 1 + (engagement / 100) * (credibility / 100)
    
    return int(followers * (reach_percentage / 100) * viral_multiplier)

def calculate_relevancy(business: BusinessDetails, influencer: Dict[str, Any]) -> float:
    """Calculate relevancy score between business category and influencer with better algorithm"""
    business_category = business.businessCategory.lower()
    influencer_category = influencer.get("category", "").lower()
    
    # Calculate category match score (similar to part of match percentage calculation)
    if business_category == influencer_category:
        category_score = 100
    elif business_category in influencer_category or influencer_category in business_category:
        category_score = 80
    else:
        # Use Jaccard similarity for partial word matching
        business_words = set(business_category.split())
        influencer_words = set(influencer_category.split())
        
        if not business_words or not influencer_words:
            category_score = 30
        else:
            intersection = len(business_words.intersection(influencer_words))
            union = len(business_words.union(influencer_words))
            category_score = (intersection / union if union > 0 else 0.3) * 100
    
    # Adjust based on engagement quality score
    engagement_quality = float(influencer.get("engagement_quality_score", 1.0))
    # Normalize if needed
    if engagement_quality > 5:
        engagement_quality = engagement_quality / 5
    
    # Calculate final relevancy with greater weight on category match
    return category_score * 0.8 + (engagement_quality * 20) * 0.2

def calculate_longevity(influencer: Dict[str, Any]) -> float:
    """Calculate longevity potential of collaboration"""
    # Get longevity score if available
    longevity_score = float(influencer.get("longevity_score", 5))
    
    # Get credibility score as it relates to longevity
    credibility = float(influencer.get("credibility_score", 50))
    
    # Normalize to 0-10 scale if needed
    if longevity_score > 10:
        longevity_score = longevity_score / 10
    
    # Weighted combination of longevity and credibility scores
    combined_score = longevity_score * 0.7 + (credibility / 10) * 0.3
        
    # Convert to 0-100 scale
    return combined_score * 10

def generate_recommendation(match_percentage: float, roi_estimate: float, business: BusinessDetails, influencer: Dict[str, Any]) -> str:
    """Generate a recommendation based on match and ROI estimates with more nuanced thresholds"""
    channel = influencer.get("channel_info", "this influencer")
    
    # More nuanced recommendation thresholds
    if match_percentage >= 80 and roi_estimate >= 300:
        return f"Exceptional Match: {channel} is perfect for {business.businessName} with very high projected ROI of {roi_estimate:.1f}%."
    elif match_percentage >= 70 and roi_estimate >= 200:
        return f"Highly Recommended: {channel} is an excellent match for {business.businessName} with strong ROI potential of {roi_estimate:.1f}%."
    elif match_percentage >= 60 and roi_estimate >= 150:
        return f"Recommended: {channel} is a strong match with solid ROI potential of {roi_estimate:.1f}%."
    elif match_percentage >= 50 and roi_estimate >= 100:
        return f"Worth Considering: {channel} is a good match with acceptable ROI potential of {roi_estimate:.1f}%."
    elif match_percentage >= 40 and roi_estimate >= 50:
        return f"Possible Option: {channel} might work for your business with modest ROI potential of {roi_estimate:.1f}%."
    else:
        return f"Not Recommended: {channel} is not an ideal match for {business.businessName} with low projected ROI of {roi_estimate:.1f}%."

def parse_follower_count(followers_str) -> float:
    """Parse follower counts like '1.2M' or '500K' to actual numbers"""
    if isinstance(followers_str, (int, float)):
        return float(followers_str)
        
    if isinstance(followers_str, str):
        followers_str = followers_str.strip().lower()
        if 'k' in followers_str:
            return float(followers_str.replace('k', '')) * 1000
        elif 'm' in followers_str:
            return float(followers_str.replace('m', '')) * 1000000
        elif 'b' in followers_str:
            return float(followers_str.replace('b', '')) * 1000000000
        else:
            return float(followers_str.replace(',', ''))
    
    return 10000  # Default

class BatchCollabRequest(BaseModel):
    business: BusinessDetails
    count: Optional[int] = 5

@app.post("/batch-collab-recommendations")
async def batch_recommendations(request: BatchCollabRequest):
    """
    Get collaboration recommendations for multiple influencers
    based on business details with improved ranking algorithm
    """
    logger.info(f"Generating batch recommendations for {request.business.businessName}")
    
    results = []
    
    # Process all influencers
    for influencer in json_data:
        influencer_username = get_username(influencer.get("channel_info", ""))
        
        # Calculate key metrics
        match_percentage = calculate_match_percentage(request.business, influencer)
        cost_estimate = calculate_cost_estimate(influencer)
        roi_estimate = calculate_roi_estimate(match_percentage, influencer, request.business)
        
        # Calculate composite score with heavy ROI emphasis
        # ROI gets 60% weight, match gets 30%, cost efficiency gets 10%
        follower_count = parse_follower_count(influencer.get("followers", "10K"))
        cost_efficiency = min(100, (follower_count / cost_estimate) / 100) if cost_estimate > 0 else 0
        
        # Scale ROI to 0-100 range for scoring purposes (cap at 500% ROI)
        scaled_roi = min(100, roi_estimate / 5)
        
        # Calculate composite score (higher is better)
        composite_score = (
            scaled_roi * 0.6 +
            match_percentage * 0.3 +
            cost_efficiency * 0.1
        )
        
        # Get recommendation text
        recommendation = generate_recommendation(match_percentage, roi_estimate, request.business, influencer)
        
        results.append({
            "username": influencer_username,
            "channel_info": influencer.get("channel_info", ""),
            "match_percentage": round(match_percentage, 1),
            "estimated_cost": round(cost_estimate, 2),
            "estimated_roi": round(roi_estimate, 2),
            "followers": influencer.get("followers", "Unknown"),
            "category": influencer.get("category", "Unknown"),
            "composite_score": round(composite_score, 1),
            "recommendation": recommendation
        })
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    
    # Return top N results
    return {"recommendations": results[:request.count]}

# ---------------------------------- Romeiro's code ends here --------------------------------------------


# API Endpoint to fetch paginated data
@app.get("/data")
def get_data(page: int = Query(1, ge=1), per_page: int = Query(200, ge=1)):
    """
    Fetch paginated JSON data.
    """
    logger.info(f"Fetching data for page {page} with {per_page} items per page")
    start = (page - 1) * per_page
    end = start + per_page
    return json_data[start:end]

# API Endpoint to fetch a specific influencer by rank
@app.get("/data/rank/{rank}")
def get_influencer_by_rank(rank: int):
    """
    Fetch influencer by rank.
    """
    logger.info(f"Fetching influencer with rank {rank}")
    influencer = next((item for item in json_data if item["rank"] == rank), None)
    
    if influencer is None:
        logger.error(f"Influencer with rank {rank} not found")
        raise HTTPException(status_code=404, detail=f"Influencer with rank {rank} not found")
    
    return influencer

# Add this endpoint to fetch all usernames
@app.get("/users")
def get_all_users():
    logger.info("Fetching all user data")
    return {"users": json_data}


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}



class Contact(BaseModel):
    phone: Optional[str]
    address: Optional[str]

class SocialMedia(BaseModel):
    instagram: Optional[str]
    twitter: Optional[str]
    linkedin: Optional[str]

class BusinessUser(BaseModel):
    businessName: str
    email: str
    businessCategory: str
    description: str
    contact: Contact              # ← Nested Contact object
    website: Optional[str]
    socialMedia: SocialMedia      # ← Nested SocialMedia object


# Endpoint to receive user data from Node
@app.post("/receive-business")
async def receive_business(user: BusinessUser):
    logger.info(f"Received business user: {user.email}")

    # Prepare influencer data as string for Gemini
    influencers_info = ""
    for influencer in json_data:
        name = influencer.get("channel_info", "Unknown")
        desc = influencer.get("description", "No description")
        influencers_info += f"Username: {name}, Description: {desc}\n"

    prompt = (
        f"You are a marketing AI assistant. Given this business description:\n\n"
        f"{user.description}\n\n"
        f"And this list of influencers:\n\n"
        f"{influencers_info}\n"
        f"Which usernames would be best to promote this business? Give me only the usernames as a list."
    )

    # Call Gemini
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        suggestions = response.text.strip()
        logger.info(f"Gemini suggestions: {suggestions}")
        
        # ✅ Return suggestions to Node.js
        return {
            "message": "Business data received and suggestions generated.",
            "suggested_influencers": suggestions
        }

    except GoogleAPICallError as e:
        logger.error(f"Gemini API error: {e.message}")
        raise HTTPException(status_code=500, detail="Gemini API failed with an error.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred.")


@app.post("/api/gemini/generate-influence")
async def generate_influence(request: InfluencerRequest):
    try:
        # Call to Google's Gemini API
        response = genai.generate_content(f"""
        Generate a world influence map for {request.influencer_name} from {request.influencer_category} category.
        Based on their home country of {request.influencer_country} and base influence score of {request.base_influence_score},
        create a JSON object with country names as keys and influence scores (0-100) as values for major countries worldwide.
        Consider cultural, geographic, linguistic proximity factors. Format as valid JSON only.
        """)
        
        # Parse the generated content
        influence_map = json.loads(response.text)
        
        return {
            "status": "success",
            "influence_map": influence_map
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "influence_map": {}
        }
        
class TrendPoint(BaseModel):
    month: str
    period: str
    quality_score: float
    engagement: float
    followers: int
    likes: int

class RegressionResult(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    predictions: List[Dict[str, Union[str, float]]]
        
@app.post("/analyze/regression/{metric}")
async def calculate_regression(metric: str, data: List[TrendPoint]):
    """Calculate linear regression for specified metric and return model details."""
    if not data or len(data) < 2:
        raise HTTPException(400, "Need at least 2 data points for regression")
    
    if metric not in ["engagement", "quality_score", "followers", "likes"]:
        raise HTTPException(400, f"Unsupported metric: {metric}")
    
    # Extract data for regression
    X = np.array(range(len(data))).reshape(-1, 1)  # X is just the indices
    y = np.array([getattr(point, metric) for point in data])
    
    # Fit regression model
    model = LinearRegression().fit(X, y)
    
    # Calculate predictions including 9 future months
    future_X = np.array(range(len(data) + 9)).reshape(-1, 1)
    predictions = model.predict(future_X)
    
    # Generate prediction results
    prediction_results = []
    for i, pred in enumerate(predictions):
        # For actual months, use existing month names
        if i < len(data):
            month = data[i].month
            period = data[i].period
        else:
            # For future months, generate month names
            # This is simplified - you might want more accurate month prediction
            last_date_parts = data[-1].period.split()
            month_idx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].index(last_date_parts[0])
            future_month_idx = (month_idx + (i - len(data) + 1)) % 12
            month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][future_month_idx]
            year = int(last_date_parts[1])
            if future_month_idx < month_idx:
                year += 1
            period = f"{month} {year}"
        
        prediction_results.append({
            "month": month,
            "period": period,
            "predicted": float(pred),
            "is_future": i >= len(data)
        })
    
    return RegressionResult(
        slope=float(model.coef_[0]),
        intercept=float(model.intercept_),
        r_squared=float(model.score(X, y)),
        predictions=prediction_results
    )
# Run the API using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
    

