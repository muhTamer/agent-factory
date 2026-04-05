import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",

  // Proxy API calls through the Next.js server to avoid CORS issues.
  // Browser → /api/concierge/* (same origin) → concierge backend
  // Browser → /api/runtime/*   (same origin) → runtime backend
  async rewrites() {
    const conciergeUrl =
      process.env.NEXT_PUBLIC_CONCIERGE_API || "http://127.0.0.1:8001";
    const runtimeUrl =
      process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:808";

    return [
      {
        source: "/api/concierge/:path*",
        destination: `${conciergeUrl}/concierge/:path*`,
      },
      {
        source: "/api/runtime/:path*",
        destination: `${runtimeUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
