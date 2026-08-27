import { defineConfig } from "vitest/config";
import { resolve } from "node:path";

export default defineConfig({
  test: {
    // Node environment on purpose: everything tested here is pure logic in lib/.
    // The geometry and status decisions are the correctness-critical part, and
    // they were deliberately kept out of components so they could be tested
    // without a DOM — which is also what makes the later visual pass a restyle.
    environment: "node",
    include: ["lib/**/*.test.ts"],
  },
  resolve: {
    alias: { "@": resolve(__dirname, ".") },
  },
});
