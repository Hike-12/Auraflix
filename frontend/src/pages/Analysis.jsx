import React from "react";
import { motion } from "framer-motion";
import Spotlight from "../components/Spotlight";
import Navbar from "../components/Navbar";
import InfluencerList from "../components/InfluencerList";
import Search from "../components/Search";
import RomFooter from "../components/RomFooter";
import Header from "../components/Header";

const Analysis = () => {
  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
      <Header/>

      <div className="relative flex-1 flex overflow-hidden bg-black/[0.96] antialiased items-center">
        <div
          className="pointer-events-none absolute inset-0 select-none opacity-80"
          style={{
            backgroundImage:
              "linear-gradient(to right, #171717 1px, transparent 1px), linear-gradient(to bottom, #171717 1px, transparent 1px)",
            backgroundSize: "35px 35px",
          }}
        />

        <Spotlight
          className="-top-40 left-0 md:-top-20 md:left-60"
          fill="white"
        />

        <div className="relative z-10 max-w-5xl mx-auto px-6 py-20 md:py-32">
          <motion.h1
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
            className="bg-gradient-to-b from-neutral-50 to-neutral-400 bg-clip-text text-center text-4xl md:text-7xl font-bricolage font-bold text-transparent"
          >
            Influencer Analysis <br /> Dashboard
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mx-auto mt-6 mb-7 max-w-2xl text-center text-base md:text-lg font-inter text-neutral-300"
          >
            Dive deep into performance metrics and audience demographics.
            Make data-driven decisions with our comprehensive analytics tools.
          </motion.p>

          <Search isEmbedded={true} />
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="mt-16"
          >
            <InfluencerList />
          </motion.div>
        </div>
      </div>

      <RomFooter/>
    </div>
  );
};

export default Analysis;