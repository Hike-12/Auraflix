import React from "react";
import { motion } from "framer-motion";
import Spotlight from "../components/Spotlight";
import Header from "../components/Header";
import RomFooter from "../components/RomFooter";
import Features from "../components/Features";
import Search from "../components/Search";

const Landing = () => {
  return (
    <div className="min-h-screen flex flex-col bg-black text-white">
      <Header />

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
            Discover the Power <br /> of Influencers
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="mx-auto mt-6 mb-7 max-w-2xl text-center text-base md:text-lg font-inter text-neutral-300"
          >
            Search, analyze, and leverage the right influencers for your brand.
            Get comprehensive reports and insights to make data-driven
            decisions.
          </motion.p>

          <Search isEmbedded={true} />
        </div>
      </div>

      <Features />
      <RomFooter />

    </div>
  );
};

export default Landing;
