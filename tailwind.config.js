/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./dashboard/templates/**/*.html", "./dashboard/static/**/*.js"],
  darkMode: "class", // << ENABLE dark mode via class toggle
  theme: {
    extend: {},
  },
  plugins: [],
};
