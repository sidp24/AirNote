/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html","./src/**/*.{ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors:{
        bg:"#0B1020", panel:"#11172A", panelAlt:"#0E1527", ink:"#EAF0FF", muted:"#9AA4B2",
        brand:{50:"#E9F0FF",100:"#CFE0FF",200:"#AACCFF",300:"#85B8FF",400:"#5AA0FF",500:"#2B82F6",600:"#2167C4",700:"#1B569F",800:"#143F73",900:"#0E2C52"}
      },
      boxShadow:{ soft:"0 10px 30px rgba(0,0,0,.35)", glow:"0 0 0 1px rgba(255,255,255,.06), 0 10px 30px rgba(56,130,246,.25)"},
      borderRadius:{ xl2:"18px"}, backdropBlur:{ xs:"2px"}
    }
  },
  plugins: [],
};