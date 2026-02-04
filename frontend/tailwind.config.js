/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"] ,
  theme: {
    extend: {
      colors: {
        pitch: {
          50: "#f1f8f4",
          100: "#dbeee2",
          200: "#b9dec9",
          700: "#0f6b3c",
          900: "#0a3b22"
        },
        left: "#1d4ed8",
        right: "#dc2626"
      }
    }
  },
  plugins: []
};
