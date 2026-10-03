import React, { useState, useRef, useEffect } from "react";
import { Terminal as FaTerminal, Send as FaPaperPlane, BarChart3 as FaChartBar, Table as FaTable, Bot, User, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { adminAPI } from "../services/api";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, LineChart, Line, Legend
} from "recharts";

const Spinner = () => (
  <div className="w-5 h-5 border-2 border-t-transparent border-[#dfc18b] rounded-full animate-spin" />
);

// Renders chart_data as a table (chart_type === "table", or any unknown chart type)
const DataTable = ({ rows }) => {
  const columns = Object.keys(rows[0] || {});
  const fmt = (v) => {
    if (typeof v === "number") return v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
    if (v === null || v === undefined) return "—";
    return String(v);
  };
  const label = (k) => k.replace(/_/g, " ");
  return (
    <div className="overflow-x-auto my-6 w-full rounded-2xl border border-white/10 shadow-2xl bg-white/5 backdrop-blur-md">
      <table className="min-w-full text-sm text-left">
        <thead className="bg-white/10 text-[#dfc18b] font-bold uppercase tracking-wider text-xs">
          <tr>
            {columns.map((c) => (
              <th key={c} className="px-6 py-4">{label(c)}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c} className="px-6 py-4 text-white/90">{fmt(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// New Chart Renderer component that works from structured state
const ChartRenderer = ({ state }) => {
  if (!state || state.chart_type === "none") return null;

  const { chart_type, chart_data } = state;
  if (!Array.isArray(chart_data) || chart_data.length === 0) return null;
  // Anything that is not bar/pie/line (e.g. "table") is shown as a table
  if (!["bar", "pie", "line"].includes(chart_type)) return <DataTable rows={chart_data} />;
  const COLORS = ['#dfc18b', '#a37c35', '#7e5d26', '#d4af37', '#b8860b'];

  return (
    <div className="h-80 w-full bg-white/5 backdrop-blur-md p-6 rounded-2xl shadow-2xl my-6 border border-white/10">
      <ResponsiveContainer width="100%" height="100%">
        {chart_type === "bar" ? (
          <BarChart data={chart_data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
            <defs>
              <linearGradient id="barGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#dfc18b" />
                <stop offset="100%" stopColor="#a37c35" />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#a8a19c', fontSize: 11}} dy={10} />
            <YAxis axisLine={false} tickLine={false} tick={{fill: '#a8a19c', fontSize: 11}} />
            <Tooltip contentStyle={{backgroundColor: '#1a1714', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#fff'}} />
            <Bar dataKey="value" fill="url(#barGradient)" radius={[6, 6, 0, 0]} barSize={32} />
          </BarChart>
        ) : chart_type === "pie" ? (
          <PieChart>
            <Pie
              data={chart_data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {chart_data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip contentStyle={{backgroundColor: '#1a1714', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#fff'}} />
            <Legend verticalAlign="bottom" height={36}/>
          </PieChart>
        ) : chart_type === "line" ? (
          <LineChart data={chart_data} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#a8a19c', fontSize: 11}} dy={10} />
            <YAxis axisLine={false} tickLine={false} tick={{fill: '#a8a19c', fontSize: 11}} />
            <Tooltip contentStyle={{backgroundColor: '#1a1714', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', color: '#fff'}} />
            <Line type="monotone" dataKey="value" stroke="#dfc18b" strokeWidth={3} dot={{ r: 4, fill: '#dfc18b' }} activeDot={{ r: 6 }} />
          </LineChart>
        ) : null}
      </ResponsiveContainer>
    </div>
  );
};

function AdminMarkdown({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        table: ({ children }) => (
          <div className="overflow-x-auto my-6 w-full rounded-2xl border border-white/10 shadow-2xl bg-white/5 backdrop-blur-md">
            <table className="min-w-full text-sm text-left">{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className="bg-white/10 text-[#dfc18b] font-bold uppercase tracking-wider text-xs">{children}</thead>,
        tbody: ({ children }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
        th: ({ children }) => <th className="px-6 py-4">{children}</th>,
        td: ({ children }) => <td className="px-6 py-4 text-white/90">{children}</td>,
        p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-[15px] opacity-90">{children}</p>,
        h3: ({ children }) => <h3 className="text-lg font-bold text-[#dfc18b] mt-6 mb-3 flex items-center gap-2"><Sparkles size={18} /> {children}</h3>,
        strong: ({ children }) => <strong className="font-bold text-white tracking-tight">{children}</strong>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

const AdminDashboard = () => {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const bottomRef = useRef(null);
  const sessionId = adminAPI.getSessionId();

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  // Load persisted history from Supabase on mount
  useEffect(() => {
    const loadHistory = async () => {
      setIsLoadingHistory(true);
      try {
        const res = await adminAPI.getHistory(sessionId);
        const stored = res.history || [];
        if (stored.length === 0) {
          setMessages([{ 
            role: "bot", 
            content: "Welcome, Commander. I am your BI Intelligence Agent. Ask me anything about your sales, products, or customer patterns.",
            state: { chart_type: "none" }
          }]);
        } else {
          // Reconstruct messages from stored history
          const reconstructed = stored.map((h) => {
            if (h.role === "user") {
              return { role: "user", content: h.content };
            } else {
              return { role: "bot", content: h.content, state: h.state || { chart_type: "none" } };
            }
          });
          setMessages(reconstructed);
        }
      } catch (e) {
        // Fallback to welcome message if history load fails
        setMessages([{ 
          role: "bot", 
          content: "Welcome, Commander. I am your BI Intelligence Agent. Ask me anything about your sales, products, or customer patterns.",
          state: { chart_type: "none" }
        }]);
      } finally {
        setIsLoadingHistory(false);
      }
    };
    loadHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleSend = async () => {
    if (!query.trim() || isTyping) return;
    const msg = query;

    setQuery("");
    setMessages(prev => [...prev, { role: "user", content: msg }]);
    setIsTyping(true);
    
    try {
      const res = await adminAPI.chat({ query: msg, session_id: sessionId });
      // res contains { reply: "...", state: { ... } }
      setMessages(prev => [...prev, { 
        role: "bot", 
        content: res.reply, 
        state: res.state 
      }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "bot", content: `**Error communicating with BI Agent:** ${e.message}`, state: { chart_type: "none" } }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="min-h-[85vh] bg-[#1a1714] flex flex-col mt-6 shadow-[0_30px_100px_rgba(0,0,0,0.6)] rounded-3xl overflow-hidden border border-white/10 relative ring-1 ring-[#dfc18b]/10">
      <div className="absolute -top-24 -left-24 w-96 h-96 bg-[#dfc18b] rounded-full blur-[120px] opacity-[0.07] pointer-events-none" />
      <div className="absolute top-1/2 -right-48 w-80 h-80 bg-[#a37c35] rounded-full blur-[100px] opacity-[0.05] pointer-events-none" />

      <div className="bg-[#1a1714]/80 backdrop-blur-xl text-white p-8 border-b border-white/5 flex items-center justify-between relative z-10 shrink-0">
        <div className="flex items-center gap-5">
          <div className="p-4 bg-gradient-to-br from-[#dfc18b]/20 to-[#a37c35]/20 rounded-2xl backdrop-blur-md border border-[#dfc18b]/20 shadow-[0_0_20px_rgba(223,193,139,0.2)]">
            <FaChartBar size={28} className="text-[#dfc18b]" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight uppercase italic flex items-center gap-2">
              BI <span className="text-[#dfc18b]">Intelligence</span>
            </h1>
            <p className="text-[#a8a19c] text-sm font-medium tracking-wide">State-driven command center for secure operations</p>
          </div>
        </div>
      </div>

      <div className="flex-grow p-6 sm:p-10 overflow-y-auto space-y-8 custom-scrollbar">
        {isLoadingHistory && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-center items-center py-12">
            <div className="flex items-center gap-3 text-[#a8a19c] text-sm font-medium tracking-wide">
              <div className="w-4 h-4 border-2 border-t-transparent border-[#dfc18b] rounded-full animate-spin" />
              <span>Restoring session...</span>
            </div>
          </motion.div>
        )}
        <AnimatePresence initial={false}>
          {messages.map((m, i) => (
            <motion.div 
              key={i} 
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ type: "spring", stiffness: 260, damping: 20 }}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className="group relative flex items-start gap-3 max-w-[90%] sm:max-w-[85%]">
                {m.role === "bot" && (
                  <div className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center shrink-0 mt-1">
                    <Bot size={16} className="text-[#dfc18b]" />
                  </div>
                )}
                <div className={`px-7 py-5 rounded-3xl shadow-xl ${
                  m.role === "user" 
                  ? "bg-gradient-to-br from-[#dfc18b] to-[#a37c35] text-[#1a1714] rounded-tr-none font-semibold" 
                  : "bg-white/5 border border-white/10 text-white rounded-tl-none backdrop-blur-md"
                }`}>
                  {m.role === "user" ? (
                    <p className="whitespace-pre-wrap leading-relaxed">{m.content}</p>
                  ) : (
                    <div>
                      <AdminMarkdown content={m.content} />
                      {m.state && <ChartRenderer state={m.state} />}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {isTyping && (
          <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="flex justify-start items-center gap-3">
            <div className="px-6 py-4 bg-white/5 border border-white/10 rounded-2xl rounded-tl-none flex items-center gap-3 backdrop-blur-md">
              <span className="text-[#dfc18b] text-xs font-black uppercase tracking-widest italic animate-pulse transition-all">Analyzing Database...</span>
            </div>
          </motion.div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="p-6 bg-[#1a1714]/80 backdrop-blur-2xl border-t border-white/5 relative z-20 shrink-0">
        <div className="max-w-4xl mx-auto relative group">
          <div className="relative flex items-center gap-4 bg-[#1a1714] border border-white/10 rounded-2xl px-6 py-2 shadow-2xl">
            <FaTerminal className="text-[#dfc18b] opacity-40 shrink-0" size={18} />
            <input 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Query data (e.g. Compare total revenue as a pie chart)..."
              className="flex-grow bg-transparent focus:outline-none text-white py-4 text-lg placeholder-white/20 font-medium"
            />
            <button onClick={handleSend} disabled={isTyping} className="bg-gradient-to-r from-[#dfc18b] to-[#a37c35] p-3 rounded-xl">
              {isTyping ? <Spinner /> : <FaPaperPlane size={14} className="text-[#1a1714]" />}
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.05); border-radius: 10px; }
      `}</style>
    </div>
  );
};

export default AdminDashboard;
