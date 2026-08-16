import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "standalone",
  outputFileTracingRoot: path.join(__dirname),

  typescript: {
    ignoreBuildErrors: true,
  },

  reactStrictMode: false,

  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
