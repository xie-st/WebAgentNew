"use client";

import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import WhatItIs from "@/components/WhatItIs";
import WhatsNew from "@/components/WhatsNew";
import Features from "@/components/Features";
import HowItWorks from "@/components/HowItWorks";
import ToolShowcase from "@/components/ToolShowcase";
import PresetShowcase from "@/components/PresetShowcase";
import ModelCompatibility from "@/components/ModelCompatibility";
import QuickStart from "@/components/QuickStart";
import Examples from "@/components/Examples";
import Community from "@/components/Community";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";

export default function HomeView() {
  return (
    <main id="main" className="min-h-screen bg-white dark:bg-[#020c08] transition-colors duration-300">
      <Navbar />
      <Hero />
      <WhatItIs />
      {/* The three ways in — the command line, Python and the server — come
          before the capability tour, because they are the first thing someone
          evaluating the framework wants and the shortest path to running it. */}
      <QuickStart />
      <WhatsNew />
      <Features />
      <HowItWorks />
      <ToolShowcase />
      <PresetShowcase />
      <ModelCompatibility />
      <Examples />
      <Community />
      <CTA />
      <Footer />
    </main>
  );
}
