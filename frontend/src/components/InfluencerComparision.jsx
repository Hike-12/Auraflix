import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { ChevronLeft, RefreshCw, Download } from "lucide-react";
import html2canvas from "html2canvas";
import jsPDF from "jspdf";
import domtoimage from "dom-to-image";

const InfluencerComparison = ({ initialInfluencers = [], onGoBack }) => {
  const [influencers, setInfluencers] = useState([]);
  const [selectedInfluencers, setSelectedInfluencers] = useState([null, null]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Parse formatted numbers (e.g. "34.7m", "367.8k") to actual numbers
  const parseFormattedNumber = (formatted) => {
    if (!formatted || typeof formatted !== "string") return 0;

    const numPart = formatted.replace(/[^0-9.]/g, "");
    const multiplier =
      formatted.endsWith("k") || formatted.endsWith("K")
        ? 1000
        : formatted.endsWith("m") || formatted.endsWith("M")
        ? 1000000
        : formatted.endsWith("b") || formatted.endsWith("B")
        ? 1000000000
        : 1;

    return parseFloat(numPart) * multiplier || 0;
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Set initial influencers if provided
        if (initialInfluencers && initialInfluencers.length > 0) {
          setSelectedInfluencers([
            initialInfluencers[0] || null,
            initialInfluencers[1] || null,
          ]);
        }

        const response = await fetch("https://influenceiq.onrender.com/users");
        if (!response.ok) {
          throw new Error("Failed to fetch user data from server.");
        }

        const data = await response.json();
        setInfluencers(data.users);
      } catch (err) {
        console.error(err);
        setError(err.message || "Something went wrong.");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [initialInfluencers]);

  const handleInfluencerSelect = (influencer, index) => {
    const newSelected = [...selectedInfluencers];
    newSelected[index] = influencer;
    setSelectedInfluencers(newSelected);
  };

  const formatLargeNumber = (num) => {
    // Safety check - if num is not a valid number, return "0"
    if (num === null || num === undefined || isNaN(num) || !isFinite(num)) {
      return "0";
    }

    num = Math.abs(num); // Handle negative numbers by taking absolute value

    if (num >= 1000000000) {
      return (num / 1000000000).toFixed(1) + "B";
    } else if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + "M";
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + "K";
    }
    return Math.round(num).toString(); // Ensure whole numbers
  };

  // Custom formatter for percentages
  const formatPercentage = (value) => `${value.toFixed(2)}%`;

  // Custom formatter for scores
  const formatScore = (value) => value.toFixed(1);

  // Prepare data for followers and likes comparison with normalized values for visualization
  const prepareAudienceData = () => {
    const data = [];

    if (selectedInfluencers[0] || selectedInfluencers[1]) {
      // For normalized chart, we create both actual values and normalized values
      const follower1 = selectedInfluencers[0]
        ? parseFormattedNumber(selectedInfluencers[0].followers)
        : 0;
      const follower2 = selectedInfluencers[1]
        ? parseFormattedNumber(selectedInfluencers[1].followers)
        : 0;
      const likes1 = selectedInfluencers[0]
        ? parseFormattedNumber(selectedInfluencers[0].avg_likes)
        : 0;
      const likes2 = selectedInfluencers[1]
        ? parseFormattedNumber(selectedInfluencers[1].avg_likes)
        : 0;
      const posts1 = selectedInfluencers[0]
        ? parseFormattedNumber(selectedInfluencers[0].posts)
        : 0;
      const posts2 = selectedInfluencers[1]
        ? parseFormattedNumber(selectedInfluencers[1].posts)
        : 0;

      // Store actual values for tooltip display
      data.push({
        name: "Followers",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]: follower1,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]: follower2,
        actualValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]: follower1,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]: follower2,
        },
      });

      data.push({
        name: "Average Likes",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]: likes1,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]: likes2,
        actualValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]: likes1,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]: likes2,
        },
      });

      data.push({
        name: "Total Posts",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]: posts1,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]: posts2,
        actualValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]: posts1,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]: posts2,
        },
      });
    }

    return data;
  };

  // Prepare data for scores comparison - normalize radar chart values to 0-100 scale
  const prepareScoreData = () => {
    const scoreData = [];

    if (selectedInfluencers[0] && selectedInfluencers[1]) {
      // Influence score (already 0-100)
      scoreData.push({
        subject: "Influence Score",
        A: selectedInfluencers[0].influence_score || 0,
        B: selectedInfluencers[1].influence_score || 0,
        fullMark: 100,
        // Store original values for tooltip display
        originalA: selectedInfluencers[0].influence_score || 0,
        originalB: selectedInfluencers[1].influence_score || 0,
      });

      // Credibility score (already 0-100)
      scoreData.push({
        subject: "Credibility",
        A: selectedInfluencers[0].credibility_score || 0,
        B: selectedInfluencers[1].credibility_score || 0,
        fullMark: 100,
        originalA: selectedInfluencers[0].credibility_score || 0,
        originalB: selectedInfluencers[1].credibility_score || 0,
      });

      // Engagement Quality (0-10 scale, normalize to 0-100)
      scoreData.push({
        subject: "Engagement Quality",
        A: (selectedInfluencers[0].engagement_quality_score || 0) * 10, // Scale to 0-100
        B: (selectedInfluencers[1].engagement_quality_score || 0) * 10, // Scale to 0-100
        fullMark: 100,
        originalA: selectedInfluencers[0].engagement_quality_score || 0,
        originalB: selectedInfluencers[1].engagement_quality_score || 0,
      });

      // Longevity (0-10 scale, normalize to 0-100)
      scoreData.push({
        subject: "Longevity",
        A: (selectedInfluencers[0].longevity_score || 0) * 10, // Scale to 0-100
        B: (selectedInfluencers[1].longevity_score || 0) * 10, // Scale to 0-100
        fullMark: 100,
        originalA: selectedInfluencers[0].longevity_score || 0,
        originalB: selectedInfluencers[1].longevity_score || 0,
      });

      // InfluenceIQ (already 0-100)
      scoreData.push({
        subject: "InfluenceIQ",
        A: selectedInfluencers[0].influenceiq_score || 0,
        B: selectedInfluencers[1].influenceiq_score || 0,
        fullMark: 100,
        originalA: selectedInfluencers[0].influenceiq_score || 0,
        originalB: selectedInfluencers[1].influenceiq_score || 0,
      });
    }

    return scoreData;
  };

  // Prepare data for engagement metrics
  const prepareEngagementData = () => {
    const data = [];

    if (selectedInfluencers[0] || selectedInfluencers[1]) {
      // Calculate engagement rate
      const getEngagementRate = (influencer) => {
        if (!influencer) return 0;
        const likes = parseFormattedNumber(influencer.avg_likes);
        const followers = parseFormattedNumber(influencer.followers);
        return followers > 0 ? (likes / followers) * 100 : 0;
      };

      data.push({
        name: "Engagement Rate (%)",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
          getEngagementRate(selectedInfluencers[0]),
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
          getEngagementRate(selectedInfluencers[1]),
      });

      data.push({
        name: "Engagement Quality",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
          selectedInfluencers[0]?.engagement_quality_score || 0,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
          selectedInfluencers[1]?.engagement_quality_score || 0,
      });
    }

    return data;
  };

  // Prepare normalized data for score breakdown
  const prepareScoreBreakdownData = () => {
    if (!selectedInfluencers[0] && !selectedInfluencers[1]) return [];

    return [
      {
        name: "Credibility",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
          selectedInfluencers[0]?.credibility_score || 0,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
          selectedInfluencers[1]?.credibility_score || 0,
        originalValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
            selectedInfluencers[0]?.credibility_score || 0,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
            selectedInfluencers[1]?.credibility_score || 0,
        },
      },
      {
        name: "Engagement Quality",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
          selectedInfluencers[0]?.engagement_quality_score || 0,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
          selectedInfluencers[1]?.engagement_quality_score || 0,
        originalValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
            selectedInfluencers[0]?.engagement_quality_score || 0,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
            selectedInfluencers[1]?.engagement_quality_score || 0,
        },
      },
      {
        name: "Longevity",
        [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
          selectedInfluencers[0]?.longevity_score || 0,
        [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
          selectedInfluencers[1]?.longevity_score || 0,
        originalValues: {
          [selectedInfluencers[0]?.channel_info || "Influencer 1"]:
            selectedInfluencers[0]?.longevity_score || 0,
          [selectedInfluencers[1]?.channel_info || "Influencer 2"]:
            selectedInfluencers[1]?.longevity_score || 0,
        },
      },
    ];
  };

  // Custom tooltip for the bar charts that shows the actual values
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black/90 p-3 border border-white/20 rounded-md backdrop-blur-md">
          <p className="font-medium text-white">{`${label}`}</p>
          {payload.map((entry, index) => {
            // Check if we have actual values stored
            const actualValue =
              entry.payload.actualValues?.[entry.name] ||
              entry.payload.originalValues?.[entry.name] ||
              entry.value;
            let displayValue = actualValue;

            // Format according to the metric type
            if (
              label === "Followers" ||
              label === "Average Likes" ||
              label === "Total Posts"
            ) {
              displayValue = formatLargeNumber(actualValue);
            } else if (label === "Engagement Rate (%)") {
              displayValue = `${actualValue.toFixed(2)}%`;
            } else if (
              label === "Engagement Quality" ||
              label === "Longevity"
            ) {
              displayValue = actualValue.toFixed(2);
            } else {
              displayValue = actualValue.toFixed(1);
            }

            return (
              <p key={index} style={{ color: entry.color }} className="text-sm">
                {`${entry.name}: ${displayValue}`}
              </p>
            );
          })}
        </div>
      );
    }
    return null;
  };

  // Custom tooltip for radar chart
  const RadarTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-black/90 p-3 border border-white/20 rounded-md backdrop-blur-md">
          <p className="font-medium text-white">{payload[0].payload.subject}</p>
          {payload.map((entry, index) => {
            // Get the original value (not normalized)
            const originalKey =
              entry.dataKey === "A" ? "originalA" : "originalB";
            const originalValue = entry.payload[originalKey];

            return (
              <p key={index} style={{ color: entry.color }} className="text-sm">
                {`${entry.name}: ${originalValue.toFixed(2)}`}
              </p>
            );
          })}
        </div>
      );
    }
    return null;
  };

  // Function to handle PDF download
  const handleDownloadPDF = () => {
    const input = document.getElementById("influencer-comparison"); // Ensure the root div has this ID

    // Capture the DOM element as a PNG image
    domtoimage
      .toPng(input, { quality: 1 }) // Use high quality
      .then((dataUrl) => {
        // Create a new PDF
        const pdf = new jsPDF("p", "mm", "a4"); // A4 size page of PDF
        const imgWidth = 210; // A4 width in mm
        const imgHeight = (imgWidth * input.offsetHeight) / input.offsetWidth; // Maintain aspect ratio

        // Add the image to the PDF
        pdf.addImage(dataUrl, "PNG", 0, 0, imgWidth, imgHeight);

        // Download the PDF
        pdf.save("influencer_comparison.pdf");
      })
      .catch((error) => {
        console.error("Error generating image:", error);
      });
  };

  return (
    <div
      id="influencer-comparison"
      className="min-h-screen flex flex-col bg-black text-white"
    >
      <div className="relative flex-1 flex overflow-hidden bg-black/[0.96] antialiased">
        <div
          className="pointer-events-none absolute inset-0 select-none"
          style={{
            backgroundImage:
              "linear-gradient(to right, #171717 1px, transparent 1px), linear-gradient(to bottom, #171717 1px, transparent 1px)",
            backgroundSize: "40px 40px",
          }}
        />

        <div className="relative z-10 w-full max-w-6xl mx-auto px-6 py-12">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-4 mb-8"
          >
            <button
              onClick={onGoBack}
              className="flex items-center gap-2 text-white/70 hover:text-white transition-colors"
            >
              <ChevronLeft size={20} />
              <span>Back</span>
            </button>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-transparent">
              Influencer Comparison
            </h1>
          </motion.div>

          {loading ? (
            <div className="flex items-center justify-center h-64">
              <RefreshCw size={32} className="animate-spin text-purple-500" />
            </div>
          ) : error ? (
            <div className="bg-red-500/20 border border-red-500/30 p-4 rounded-md text-red-300">
              {error}
            </div>
          ) : (
            <>
              {/* Influencer Selection */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10"
              >
                {[0, 1].map((index) => (
                  <div
                    key={index}
                    className="bg-white/5 p-6 rounded-xl border border-white/10"
                  >
                    <h3 className="text-xl font-medium mb-4 text-white/90">
                      Select Influencer {index + 1}
                    </h3>
                    <select
                      className="w-full p-3 bg-black/50 border border-white/20 rounded-lg text-white"
                      value={selectedInfluencers[index]?.channel_info || ""}
                      onChange={(e) => {
                        const selected = influencers.find(
                          (inf) => inf.channel_info === e.target.value
                        );
                        handleInfluencerSelect(selected, index);
                      }}
                    >
                      <option value="">-- Select an influencer --</option>
                      {influencers.map((inf) => (
                        <option key={inf.channel_info} value={inf.channel_info}>
                          @{inf.channel_info}
                        </option>
                      ))}
                    </select>

                    {selectedInfluencers[index] && (
                      <div className="mt-4">
                        <div className="flex items-center gap-3">
                          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-xl font-bold">
                            {selectedInfluencers[index].channel_info
                              .charAt(0)
                              .toUpperCase()}
                          </div>
                          <div>
                            <p className="font-medium">
                              @{selectedInfluencers[index].channel_info}
                            </p>
                            <p className="text-sm text-white/60">
                              Rank #{selectedInfluencers[index].rank} •
                              {selectedInfluencers[index].country
                                ? ` ${selectedInfluencers[index].country}`
                                : " Unknown"}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </motion.div>

              {/* Comparison Graphs */}
              {(selectedInfluencers[0] || selectedInfluencers[1]) && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-8"
                >
                  {/* Followers, Likes and Posts as Separate Graphs */}
                  <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                    <h3 className="text-xl font-medium mb-6 text-white/90">
                      Audience & Content Metrics
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {/* Followers Graph */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Followers
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Followers",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]: selectedInfluencers[0]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[0].followers
                                    )
                                  : 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]: selectedInfluencers[1]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[1].followers
                                    )
                                  : 0,
                                actualValues: {
                                  [selectedInfluencers[0]?.channel_info ||
                                  "Influencer 1"]: selectedInfluencers[0]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[0].followers
                                      )
                                    : 0,
                                  [selectedInfluencers[1]?.channel_info ||
                                  "Influencer 2"]: selectedInfluencers[1]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[1].followers
                                      )
                                    : 0,
                                },
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, "auto"]}
                              tickFormatter={(value) =>
                                formatLargeNumber(value)
                              }
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#8884d8"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#82ca9d"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Average Likes Graph */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Average Likes
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Avg Likes",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]: selectedInfluencers[0]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[0].avg_likes
                                    )
                                  : 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]: selectedInfluencers[1]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[1].avg_likes
                                    )
                                  : 0,
                                actualValues: {
                                  [selectedInfluencers[0]?.channel_info ||
                                  "Influencer 1"]: selectedInfluencers[0]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[0].avg_likes
                                      )
                                    : 0,
                                  [selectedInfluencers[1]?.channel_info ||
                                  "Influencer 2"]: selectedInfluencers[1]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[1].avg_likes
                                      )
                                    : 0,
                                },
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, "auto"]}
                              tickFormatter={(value) =>
                                formatLargeNumber(value)
                              }
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#8884d8"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#82ca9d"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Total Posts Graph */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Total Posts
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Posts",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]: selectedInfluencers[0]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[0].posts
                                    )
                                  : 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]: selectedInfluencers[1]
                                  ? parseFormattedNumber(
                                      selectedInfluencers[1].posts
                                    )
                                  : 0,
                                actualValues: {
                                  [selectedInfluencers[0]?.channel_info ||
                                  "Influencer 1"]: selectedInfluencers[0]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[0].posts
                                      )
                                    : 0,
                                  [selectedInfluencers[1]?.channel_info ||
                                  "Influencer 2"]: selectedInfluencers[1]
                                    ? parseFormattedNumber(
                                        selectedInfluencers[1].posts
                                      )
                                    : 0,
                                },
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, "auto"]}
                              tickFormatter={(value) =>
                                formatLargeNumber(value)
                              }
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#8884d8"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#82ca9d"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                  {/* Score Metrics (Radar Chart) - normalized to 0-100 scale */}
                  {selectedInfluencers[0] && selectedInfluencers[1] && (
                    <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                      <h3 className="text-xl font-medium mb-6 text-white/90">
                        Score Comparison
                      </h3>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <RadarChart
                            outerRadius="70%"
                            data={prepareScoreData()}
                          >
                            <PolarGrid stroke="#444" />
                            <PolarAngleAxis dataKey="subject" stroke="#aaa" />
                            <PolarRadiusAxis
                              angle={30}
                              domain={[0, 100]}
                              stroke="#aaa"
                            />
                            <Radar
                              name={selectedInfluencers[0]?.channel_info}
                              dataKey="A"
                              stroke="#8884d8"
                              fill="#8884d8"
                              fillOpacity={0.6}
                            />
                            <Radar
                              name={selectedInfluencers[1]?.channel_info}
                              dataKey="B"
                              stroke="#82ca9d"
                              fill="#82ca9d"
                              fillOpacity={0.6}
                            />
                            <Legend />
                            <Tooltip content={<RadarTooltip />} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="mt-2 text-center text-white/60 text-sm">
                        Note: All metrics are normalized to a 0-100 scale for
                        visualization. Hover for actual values.
                      </div>
                    </div>
                  )}

                  {/* Engagement Metrics as separate graphs */}
                  <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                    <h3 className="text-xl font-medium mb-6 text-white/90">
                      Engagement Metrics
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Engagement Rate Graph */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Engagement Rate (%)
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Engagement Rate",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]: selectedInfluencers[0]
                                  ? (parseFormattedNumber(
                                      selectedInfluencers[0].avg_likes
                                    ) /
                                      parseFormattedNumber(
                                        selectedInfluencers[0].followers
                                      )) *
                                    100
                                  : 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]: selectedInfluencers[1]
                                  ? (parseFormattedNumber(
                                      selectedInfluencers[1].avg_likes
                                    ) /
                                      parseFormattedNumber(
                                        selectedInfluencers[1].followers
                                      )) *
                                    100
                                  : 0,
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, "auto"]}
                              tickFormatter={(value) => `${value.toFixed(2)}%`}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#ff84d8"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#f7ca9d"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Engagement Quality Graph */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Engagement Quality (0-10)
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Quality",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]:
                                  selectedInfluencers[0]
                                    ?.engagement_quality_score || 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]:
                                  selectedInfluencers[1]
                                    ?.engagement_quality_score || 0,
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, 10]}
                              tickCount={6}
                              tickFormatter={(value) => value.toFixed(1)}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#ff84d8"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#f7ca9d"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Performance Score Breakdown - separate graphs for each score */}
                  <div className="bg-white/5 p-6 rounded-xl border border-white/10">
                    <h3 className="text-xl font-medium mb-6 text-white/90">
                      Performance Score Breakdown
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      {/* Credibility Score */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Credibility (0-100)
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Credibility",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]:
                                  selectedInfluencers[0]?.credibility_score ||
                                  0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]:
                                  selectedInfluencers[1]?.credibility_score ||
                                  0,
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, 100]}
                              tickCount={6}
                              tickFormatter={(value) => value.toFixed(0)}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#fc766a"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#5b84b1"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Engagement Quality Score */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Engagement Quality (0-10)
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Quality",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]:
                                  selectedInfluencers[0]
                                    ?.engagement_quality_score || 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]:
                                  selectedInfluencers[1]
                                    ?.engagement_quality_score || 0,
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, 10]}
                              tickCount={6}
                              tickFormatter={(value) => value.toFixed(1)}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#fc766a"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#5b84b1"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Longevity Score */}
                      <div className="h-80">
                        <h4 className="text-center text-white/80 mb-2">
                          Longevity (0-10)
                        </h4>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={[
                              {
                                name: "Longevity",
                                [selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"]:
                                  selectedInfluencers[0]?.longevity_score || 0,
                                [selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"]:
                                  selectedInfluencers[1]?.longevity_score || 0,
                              },
                            ]}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 40,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#333"
                            />
                            <XAxis dataKey="name" stroke="#aaa" />
                            <YAxis
                              stroke="#aaa"
                              domain={[0, 10]}
                              tickCount={6}
                              tickFormatter={(value) => value.toFixed(1)}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <Legend
                              verticalAlign="top"
                              wrapperStyle={{ paddingBottom: "20px" }}
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[0]?.channel_info ||
                                "Influencer 1"
                              }
                              fill="#fc766a"
                              name={
                                selectedInfluencers[0]?.channel_info ||
                                "Not Selected"
                              }
                            />
                            <Bar
                              dataKey={
                                selectedInfluencers[1]?.channel_info ||
                                "Influencer 2"
                              }
                              fill="#5b84b1"
                              name={
                                selectedInfluencers[1]?.channel_info ||
                                "Not Selected"
                              }
                            />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Download Button */}
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-center mt-8"
                  >
                    <button
                      onClick={handleDownloadPDF}
                      className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-blue-600 px-6 py-3 rounded-lg text-white font-medium hover:from-purple-700 hover:to-blue-700 transition-all"
                    >
                      <Download size={18} />
                      Download Comparison Report
                    </button>
                  </motion.div>
                </motion.div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default InfluencerComparison;
