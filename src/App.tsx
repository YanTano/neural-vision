import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI } from '@google/genai';
import { Camera, Square, Play, Music, Loader2, AlertCircle, Key, Activity, Cpu, ScanFace, Info, X, Palette, Sparkles, Sliders, Users, ShieldCheck, UserCheck, RefreshCw, Eye, Thermometer, Heart, Radio, Mic, Settings, Layers, Smartphone, FileText, Car, HelpCircle, Monitor, Wifi, BatteryCharging, Zap, Target, Volume2, VolumeX, Check, ChevronRight, Compass, Briefcase, MessageSquare, Award, ThumbsUp, CheckCircle2, BarChart2 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import * as tf from '@tensorflow/tfjs';

export interface BiometricSubject {
  id: string;
  label: string;
  emotion: string;
  matchScore: number;
  hr: number;
  type: 'face' | 'body';
  color: string;
  blendshapes?: { smile: number; frown: number; mouthOpen: number; browRaise: number; eyeBlink: number; pucker: number };
}
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'motion/react';
import * as Tone from 'tone';

let hoverSynth: Tone.Synth | null = null;

async function initAudio() {
  if (Tone.context.state !== 'running') {
    await Tone.start().catch(() => {});
  }
  if (!hoverSynth) {
    hoverSynth = new Tone.Synth({
      oscillator: { type: 'sine' },
      envelope: { attack: 0.01, decay: 0.1, sustain: 0, release: 0.01 }
    }).toDestination();
    hoverSynth.volume.value = -15;
  }
}

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

interface SmoothedBox {
  x: number;
  y: number;
  width: number;
  height: number;
  class: string;
  score: number;
  opacity: number;
  labelX: number;
  labelY: number;
}

function MetricSparkline({ data, color = '#00f0ff' }: { data: number[]; color?: string }) {
  const chartData = (!data || data.length === 0) 
    ? [50, 55, 60] 
    : data.length === 1 
      ? [Math.max(10, data[0] - 15), data[0]] 
      : data;

  const min = 0;
  const max = 100;
  const points = chartData.map((val, idx) => {
    const x = (idx / (chartData.length - 1)) * 100;
    const y = 20 - ((Math.min(max, Math.max(min, val)) - min) / (max - min)) * 16 - 2;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(' L ')}`;
  const lastPoint = points[points.length - 1].split(',');

  return (
    <svg className="w-16 sm:w-20 h-4.5 shrink-0 overflow-visible" viewBox="0 0 100 20">
      <defs>
        <linearGradient id={`grad-${color.replace(/[^a-zA-Z0-9]/g, '')}`} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <path
        d={`${pathD} L 100,20 L 0,20 Z`}
        fill={`url(#grad-${color.replace(/[^a-zA-Z0-9]/g, '')})`}
      />
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="drop-shadow-[0_0_6px_rgba(0,240,255,0.6)]"
      />
      {chartData.map((val, idx) => {
        const x = (idx / (chartData.length - 1)) * 100;
        const y = 20 - ((Math.min(max, Math.max(min, val)) - min) / (max - min)) * 16 - 2;
        return (
          <circle
            key={idx}
            cx={x}
            cy={y}
            r="1.5"
            fill={color}
          />
        );
      })}
      <circle
        cx={lastPoint[0]}
        cy={lastPoint[1]}
        r="3"
        fill={color}
        className="animate-pulse"
      />
    </svg>
  );
}

class PCMPlayer {
  audioContext: AudioContext;
  nextStartTime: number;

  constructor(sampleRate: number = 48000) {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate });
    this.nextStartTime = this.audioContext.currentTime;
  }

  playChunk(base64Data: string) {
    const binaryString = atob(base64Data);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // 16-bit PCM stereo
    const int16Array = new Int16Array(bytes.buffer);
    const numSamples = int16Array.length / 2;
    const leftChannel = new Float32Array(numSamples);
    const rightChannel = new Float32Array(numSamples);
    
    for (let i = 0; i < numSamples; i++) {
      leftChannel[i] = int16Array[i * 2] / 32768.0;
      rightChannel[i] = int16Array[i * 2 + 1] / 32768.0;
    }

    const audioBuffer = this.audioContext.createBuffer(2, numSamples, this.audioContext.sampleRate);
    audioBuffer.getChannelData(0).set(leftChannel);
    audioBuffer.getChannelData(1).set(rightChannel);

    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);

    const currentTime = this.audioContext.currentTime;
    if (this.nextStartTime < currentTime) {
      this.nextStartTime = currentTime + 0.05;
    }

    source.start(this.nextStartTime);
    this.nextStartTime += audioBuffer.duration;
  }

  stop() {
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }
}

class ProceduralMusicEngine {
  audioContext: AudioContext;
  isPlaying: boolean = false;
  currentVibe: string = 'minimalist ambient drone, quiet';
  targetVibe: string = 'minimalist ambient drone, quiet';
  vibeBlend: number = 1.0;
  nextNoteTime: number = 0;
  timerID: number | null = null;
  
  // Scales (intervals from root)
  scales: Record<string, number[]> = {
    major: [0, 2, 4, 5, 7, 9, 11],
    minor: [0, 2, 3, 5, 7, 8, 10],
    pentatonic: [0, 2, 4, 7, 9],
    cyberpunk: [0, 3, 7, 8, 10], // Phrygian dominant-ish
    drone: [0, 7], // Just roots and fifths
    melancholic: [0, 2, 3, 7, 8], // Minor pentatonic-ish
    dissonant: [0, 1, 6, 7, 11], // For fear/disgust
    tribal: [0, 3, 5, 7, 10] // Minor pentatonic
  };

  constructor() {
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  }

  setVibe(vibe: string) {
    if (this.targetVibe !== vibe) {
      if (this.vibeBlend >= 1.0) {
        this.currentVibe = this.targetVibe;
      }
      this.targetVibe = vibe;
      this.vibeBlend = 0.0;
    }
  }

  start() {
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    this.isPlaying = true;
    this.nextNoteTime = this.audioContext.currentTime + 0.1;
    this.scheduleNext();
  }

  stop() {
    this.isPlaying = false;
    if (this.timerID !== null) {
      clearTimeout(this.timerID);
      this.timerID = null;
    }
    if (this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
  }

  playNote(freq: number, type: OscillatorType, duration: number, vol: number, attack: number, time: number) {
    if (this.audioContext.state === 'closed') return;
    
    // Create multiple oscillators for a thicker soundscape
    const numOscs = 4;
    const masterGain = this.audioContext.createGain();
    masterGain.connect(this.audioContext.destination);
    
    const now = time;
    masterGain.gain.setValueAtTime(0, now);
    masterGain.gain.linearRampToValueAtTime(vol, now + attack);
    masterGain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    // Add a subtle reverb effect using a convolver or just delay
    const delay = this.audioContext.createDelay();
    delay.delayTime.value = 0.33;
    const feedback = this.audioContext.createGain();
    feedback.gain.value = 0.4;
    delay.connect(feedback);
    feedback.connect(delay);
    delay.connect(masterGain);

    for (let i = 0; i < numOscs; i++) {
      const osc = this.audioContext.createOscillator();
      const filter = this.audioContext.createBiquadFilter();
      
      osc.type = i % 2 === 0 ? type : 'sine';
      osc.frequency.value = freq * (1 + (i * 0.008)); // Slight detune
      
      filter.type = 'lowpass';
      filter.frequency.value = freq * 2;
      filter.frequency.linearRampToValueAtTime(freq * 6, now + attack);
      filter.frequency.linearRampToValueAtTime(freq * 1.5, now + duration);
      
      osc.connect(filter);
      filter.connect(masterGain);
      filter.connect(delay); // Send to delay for space
      
      osc.start(now);
      osc.stop(now + duration);
    }
  }

  getTempoForVibe(vibe: string): number {
    if (vibe.includes('tribal') || vibe.includes('rhythmic')) return 100;
    if (vibe.includes('cyberpunk') || vibe.includes('electronic')) return 60;
    return 40;
  }

  scheduleNext() {
    if (!this.isPlaying) return;
    
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
    
    while (this.nextNoteTime < this.audioContext.currentTime + 0.5) {
      if (this.vibeBlend < 1.0) {
        this.vibeBlend += 0.02; // crossfade over 50 notes for a much smoother transition
        if (this.vibeBlend > 1.0) this.vibeBlend = 1.0;
      }

      if (this.vibeBlend < 1.0) {
        // Equal power crossfade for smoother audio blending
        const currentWeight = Math.cos(this.vibeBlend * 0.5 * Math.PI);
        const targetWeight = Math.sin(this.vibeBlend * 0.5 * Math.PI);
        this.generateTickForVibe(this.currentVibe, currentWeight, this.nextNoteTime);
        this.generateTickForVibe(this.targetVibe, targetWeight, this.nextNoteTime);
      } else {
        this.generateTickForVibe(this.targetVibe, 1.0, this.nextNoteTime);
      }
      
      // Smoothly interpolate tempo
      const currentTempo = this.getTempoForVibe(this.currentVibe);
      const targetTempo = this.getTempoForVibe(this.targetVibe);
      const tempo = currentTempo * (1 - this.vibeBlend) + targetTempo * this.vibeBlend;
      
      const secondsPerBeat = 60.0 / tempo;
      this.nextNoteTime += secondsPerBeat; // Quarter notes
    }
    
    this.timerID = window.setTimeout(() => this.scheduleNext(), 50);
  }

  generateTickForVibe(vibe: string, weight: number, time: number) {
    if (weight <= 0.01) return;
    
    const isCyberpunk = vibe.includes('cyberpunk') || vibe.includes('electronic');
    const isTribal = vibe.includes('tribal') || vibe.includes('rhythmic') || vibe.includes('happy');
    const isAcoustic = vibe.includes('acoustic') || vibe.includes('guitar');
    const isAmbient = vibe.includes('ambient') || vibe.includes('drone');
    const isSad = vibe.includes('sad') || vibe.includes('melancholy');
    const isTense = vibe.includes('angry') || vibe.includes('fear') || vibe.includes('disgust');
    
    let scale = this.scales.pentatonic;
    let baseNote = 48; // C3
    let oscType: OscillatorType = 'sine';
    let vol = 0.08;
    let duration = 6.0; // Longer durations for soundscape
    let attack = 3.0;

    if (isCyberpunk) {
      scale = this.scales.cyberpunk;
      baseNote = 36; // C2
      oscType = 'sawtooth';
      vol = 0.04;
      duration = 4.0;
      attack = 2.0;
    } else if (isTribal) {
      scale = this.scales.tribal;
      baseNote = 43; // G2
      oscType = 'square';
      vol = 0.06;
      duration = 1.5;
      attack = 0.1;
    } else if (isSad) {
      scale = this.scales.melancholic;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 8.0;
      attack = 4.0;
    } else if (isTense) {
      scale = this.scales.dissonant;
      baseNote = 36;
      oscType = 'sawtooth';
      vol = 0.05;
      duration = 5.0;
      attack = 1.5;
    } else if (isAcoustic) {
      scale = this.scales.major;
      baseNote = 48;
      oscType = 'sine';
      vol = 0.08;
      duration = 5.0;
      attack = 2.0;
    } else if (isAmbient) {
      scale = this.scales.drone;
      baseNote = 36;
      oscType = 'sine';
      vol = 0.12;
      duration = 10.0;
      attack = 5.0;
    }

    vol *= weight; // Apply crossfade weight

    // Randomly play a note from the scale
    if (Math.random() > 0.2) {
      const noteIndex = scale[Math.floor(Math.random() * scale.length)];
      const freq = 440 * Math.pow(2, (baseNote + noteIndex - 69) / 12);
      this.playNote(freq, oscType, duration, vol, attack, time);
    }
    
    // Add a bass drone
    if (Math.random() > 0.5) {
      const bassFreq = 440 * Math.pow(2, (baseNote - 12 - 69) / 12);
      this.playNote(bassFreq, 'sine', duration * 2, vol * 1.5, attack * 2, time);
    }
  }
}

const VIBE_MAP: Record<string, string> = {
  person: "ethereal ambient drone, calm",
  'cell phone': "cyberpunk synthwave, electronic",
  laptop: "cyberpunk synthwave, electronic",
  tv: "cyberpunk synthwave, electronic",
  cup: "coffee shop jazz, chill acoustic",
  bottle: "coffee shop jazz, chill acoustic",
  bowl: "coffee shop jazz, chill acoustic",
  cat: "playful acoustic guitar, happy melody",
  dog: "playful acoustic guitar, happy melody",
  bird: "playful acoustic guitar, happy melody",
  car: "driving rock beat, fast tempo",
  bus: "driving rock beat, fast tempo",
  truck: "driving rock beat, fast tempo",
  chair: "ambient drone, relaxing",
  couch: "ambient drone, relaxing",
  bed: "ambient drone, relaxing",
  'potted plant': "ethereal flute, ambient nature",
  book: "classical piano, focused",
};

function getVibeForObjects(objects: string[]) {
  if (objects.length === 0) return "minimalist ambient drone, quiet";
  
  const vibes = new Set<string>();
  for (const obj of objects) {
    if (VIBE_MAP[obj]) {
      vibes.add(VIBE_MAP[obj]);
    } else {
      vibes.add("chill lofi beat");
    }
  }
  
  return Array.from(vibes).slice(0, 2).join(", ");
}

const getVibeFromGemini = async (objects: string[], emotion: string): Promise<string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY });
    const prompt = `You are a soundscape generator. Based on the following scene, output a 3-5 word ambient soundscape description (e.g., 'tribal rhythmic drone', 'cyberpunk electronic drone' or 'melancholy acoustic ambient'). Do not include any other text. Never output 'pop', 'upbeat', or 'energetic'. Everything must be ambient, but based on the expression. Scene: A person is feeling ${emotion} and the following objects are visible: ${objects.length > 0 ? objects.join(', ') : 'none'}.`;
    const response = await ai.models.generateContent({
      model: 'gemini-flash-lite-latest',
      contents: prompt,
    });
    return response.text?.trim() || "ambient drone, relaxing";
  } catch (e: any) {
    console.warn("Gemini API error (falling back to local vibe map):", e.message || e);
    return getVibeForObjects(objects) + `, ${emotion} mood`;
  }
};

const CLASS_COLORS: Record<string, string> = {
  person: '#00f3ff',        // Cyber Cyan
  'cell phone': '#ff007f',   // Neon Magenta
  laptop: '#a855f7',         // Bright Purple
  tv: '#8b5cf6',             // Violet
  cup: '#f59e0b',            // Amber Gold
  bottle: '#f59e0b',         // Amber Gold
  bowl: '#f59e0b',           // Amber Gold
  cat: '#10b981',            // Emerald Green
  dog: '#10b981',            // Emerald Green
  bird: '#34d399',           // Mint
  car: '#3b82f6',            // Electric Blue
  bus: '#3b82f6',            // Electric Blue
  truck: '#3b82f6',          // Electric Blue
  chair: '#f43f5e',          // Coral Rose
  couch: '#f43f5e',          // Coral Rose
  bed: '#f43f5e',            // Coral Rose
  'potted plant': '#22c55e', // Vivid Green
  book: '#eab308',           // Bright Yellow
};

function getClassColor(className: string): string {
  const lower = className.toLowerCase();
  if (CLASS_COLORS[lower]) return CLASS_COLORS[lower];
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 90%, 65%)`;
}

const EMOTION_COLORS: Record<string, { main: string; glow: string; label: string }> = {
  happy: { main: '#fbbf24', glow: 'rgba(251, 191, 36, 0.6)', label: 'text-amber-400' },
  sadness: { main: '#818cf8', glow: 'rgba(129, 140, 248, 0.6)', label: 'text-indigo-400' },
  surprised: { main: '#22d3ee', glow: 'rgba(34, 211, 238, 0.6)', label: 'text-cyan-400' },
  angry: { main: '#f87171', glow: 'rgba(248, 113, 113, 0.6)', label: 'text-rose-400' },
  fear: { main: '#c084fc', glow: 'rgba(192, 132, 252, 0.6)', label: 'text-purple-400' },
  disgust: { main: '#34d399', glow: 'rgba(52, 211, 153, 0.6)', label: 'text-emerald-400' },
  neutral: { main: '#38bdf8', glow: 'rgba(56, 189, 248, 0.6)', label: 'text-sky-400' },
};

function getEmotionColor(emotion: string) {
  return EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral;
}

interface ColorModeOption {
  id: string;
  name: string;
  desc: string;
  cssFilter: string;
  accentColor: string;
  badgeBg: string;
  textColor: string;
}

const CAMERA_COLOR_MODES: ColorModeOption[] = [
  { id: 'cyber', name: 'Cyber Neon', desc: 'Futuristic cyan and electric blue tint', cssFilter: 'hue-rotate-180 contrast-125 saturate-200', accentColor: '#00f3ff', badgeBg: 'bg-cyan-500/20 border-cyan-500/50', textColor: 'text-cyan-400' },
  { id: 'vivid', name: 'Vivid Natural', desc: 'High contrast and rich realistic tones', cssFilter: 'saturate-150 contrast-110 brightness-105', accentColor: '#10b981', badgeBg: 'bg-emerald-500/20 border-emerald-500/50', textColor: 'text-emerald-400' },
  { id: 'thermal', name: 'Thermal Heat', desc: 'Infrared heat spectrum visualization', cssFilter: 'contrast-200 saturate-250 hue-rotate-90', accentColor: '#f43f5e', badgeBg: 'bg-rose-500/20 border-rose-500/50', textColor: 'text-rose-400' },
  { id: 'sunset', name: 'Warm Sunset', desc: 'Warm amber and twilight sunset tones', cssFilter: 'sepia-50 saturate-200 hue-rotate-330 brightness-105', accentColor: '#f59e0b', badgeBg: 'bg-amber-500/20 border-amber-500/50', textColor: 'text-amber-400' },
  { id: 'emerald', name: 'Matrix Emerald', desc: 'Classic hacker terminal green visor filter', cssFilter: 'sepia-100 hue-rotate-90 saturate-200 contrast-125', accentColor: '#22c55e', badgeBg: 'bg-green-500/20 border-green-500/50', textColor: 'text-green-400' },
  { id: 'mono', name: 'Cyber Monochrome', desc: 'High contrast monochrome tactical view', cssFilter: 'grayscale contrast-125', accentColor: '#a1a1aa', badgeBg: 'bg-zinc-500/20 border-zinc-500/50', textColor: 'text-zinc-300' },
];

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraColorMode, setCameraColorMode] = useState<string>('cyber');
  const [status, setStatus] = useState('Loading Object Detection Model...');
  const [currentPrompt, setCurrentPrompt] = useState('Waiting for camera...');
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [infoMsg, setInfoMsg] = useState<string | null>(null);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [activeAffectiveTab, setActiveAffectiveTab] = useState<string>('all');
  const [isRecording, setIsRecording] = useState(false);
  const [isScanActive, setIsScanActive] = useState(true);
  const [isVoiceActive, setIsVoiceActive] = useState(true);
  const [isDataOpen, setIsDataOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [currentTimeStr, setCurrentTimeStr] = useState<string>('10:42 PM');
  const [sysMetrics, setSysMetrics] = useState({ cpu: 42, gpu: 68, ram: 58, fps: 60, temp: 45 });
  const [mobileHudTab, setMobileHudTab] = useState<'all' | 'affective' | 'reticle' | 'objects'>('all');

  // Extended Settings & Compass Navigation States
  const [showFaceMesh, setShowFaceMesh] = useState(true);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showBiometrics, setShowBiometrics] = useState(true);
  const [detectionThreshold, setDetectionThreshold] = useState(0.5);
  const [masterVolume, setMasterVolume] = useState(80);
  const [musicVibeStyle, setMusicVibeStyle] = useState('ambient');
  const [compassHeading, setCompassHeading] = useState(45);
  const [isGyroActive, setIsGyroActive] = useState(true);

  // AI Mock Interview Mode States
  const [isInterviewOpen, setIsInterviewOpen] = useState(false);
  const [interviewStep, setInterviewStep] = useState(0);
  const [userAnswerInput, setUserAnswerInput] = useState('');
  const [isMicListening, setIsMicListening] = useState(false);
  const [micVolume, setMicVolume] = useState(0);
  const [micStatusText, setMicStatusText] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isInterviewerSpeaking, setIsInterviewerSpeaking] = useState(false);
  const [isEvaluatingAnswer, setIsEvaluatingAnswer] = useState(false);
  const [interviewHistory, setInterviewHistory] = useState<Array<{
    question: string;
    answer: string;
    verbalScore: number;
    nonVerbalScore: number;
    emotionAtAnswer: string;
    feedback: string;
    tips: string;
  }>>([]);
  const [showFinalReport, setShowFinalReport] = useState(false);
  const [interviewBiometricsLog, setInterviewBiometricsLog] = useState<Array<{
    step: number;
    emotion: string;
    confidence: number;
    stress: number;
    positivity: number;
  }>>([]);

  const [metricHistory, setMetricHistory] = useState<Record<string, number[]>>({
    FOCUS: [62, 65, 68, 70, 72, 75, 78, 80, 84, 88],
    ATTENTION: [70, 72, 75, 74, 78, 82, 80, 85, 83, 89],
    STRESS: [35, 30, 28, 25, 22, 20, 18, 16, 15, 12],
    CONFIDENCE: [58, 60, 64, 68, 72, 76, 80, 82, 85, 90],
    ENERGY: [50, 54, 58, 62, 65, 68, 70, 72, 75, 80]
  });

  const recognitionRef = useRef<any>(null);
  const baseAnswerTextRef = useRef<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioAnalyserRef = useRef<AnalyserNode | null>(null);
  const audioAnimFrameRef = useRef<number | null>(null);
  const audioStreamRef = useRef<MediaStream | null>(null);

  const showFaceMeshRef = useRef(true);
  const showBoundingBoxesRef = useRef(true);
  const detectionThresholdRef = useRef(0.5);

  useEffect(() => { showFaceMeshRef.current = showFaceMesh; }, [showFaceMesh]);
  useEffect(() => { showBoundingBoxesRef.current = showBoundingBoxes; }, [showBoundingBoxes]);
  useEffect(() => { detectionThresholdRef.current = detectionThreshold; }, [detectionThreshold]);

  // Orientation & Compass Effect
  useEffect(() => {
    let interval: any;
    const handleOrientation = (e: DeviceOrientationEvent) => {
      if (!isGyroActive) return;
      let heading = 0;
      if ((e as any).webkitCompassHeading !== undefined && (e as any).webkitCompassHeading !== null) {
        heading = (e as any).webkitCompassHeading;
      } else if (e.alpha !== null) {
        heading = (360 - e.alpha) % 360;
      } else {
        return;
      }
      setCompassHeading(Math.round(heading));
    };

    if (window.DeviceOrientationEvent && isGyroActive) {
      window.addEventListener('deviceorientation', handleOrientation);
    }

    interval = setInterval(() => {
      if (!isGyroActive) return;
      setCompassHeading(prev => {
        const delta = (Math.random() - 0.5) * 1.5;
        return Math.round((prev + delta + 360) % 360);
      });
    }, 1500);

    return () => {
      if (window.DeviceOrientationEvent) {
        window.removeEventListener('deviceorientation', handleOrientation);
      }
      clearInterval(interval);
    };
  }, [isGyroActive]);

  useEffect(() => {
    try {
      const dest = Tone.getDestination();
      if (dest) {
        dest.volume.value = Tone.gainToDb(Math.max(0.01, masterVolume / 100));
      }
    } catch (e) {
      // Audio context not initialized yet
    }
  }, [masterVolume]);

  const getHeadingCardinal = (deg: number) => {
    const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(((deg % 360) + 360) % 360 / 45) % 8;
    return dirs[index];
  };

  const [consoleState, setConsoleState] = useState({
    emotion: 'neutral',
    objects: [] as string[],
    blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 },
    subjects: [] as BiometricSubject[]
  });
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const faceCanvasRef = useRef<HTMLCanvasElement>(null);
  const interviewVideoRef = useRef<HTMLVideoElement>(null);
  const interviewCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionRef = useRef<any>(null);
  const playerRef = useRef<PCMPlayer | null>(null);

  // Sync camera stream and detection loop when AI Interview modal opens
  useEffect(() => {
    if (isInterviewOpen) {
      if (!isCameraActive && isModelLoaded) {
        startSession();
      } else if (!isPlayingRef.current && streamRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);
      }
      if (interviewVideoRef.current && streamRef.current) {
        interviewVideoRef.current.srcObject = streamRef.current;
        interviewVideoRef.current.play().catch(e => console.error("Interview video play error:", e));
      }
    }
  }, [isInterviewOpen, isCameraActive, isModelLoaded]);
  
  const objectModelRef = useRef<cocoSsd.ObjectDetection | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const isPlayingRef = useRef(false);
  const lastPromptRef = useRef<string>("");
  const lastStateRef = useRef<string>("");
  const pendingStateRef = useRef<string | null>(null);
  const vibeTimeoutRef = useRef<any>(null);
  const lastStateUpdateTimeRef = useRef<number>(0);
  const detectLoopRef = useRef<number | null>(null);
  const smoothedBoxesRef = useRef<Map<string, SmoothedBox>>(new Map());
  const smoothedBlendshapesRef = useRef({ smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 });
  const smoothedSubjectBlendshapesRef = useRef<Map<number, { smile: number; frown: number; mouthOpen: number; browRaise: number; eyeBlink: number; pucker: number }>>(new Map());

  const playHoverSound = () => {
    try {
      initAudio();
      if (!hoverSynth || Tone.context.state !== 'running') return;
      
      const now = Tone.now();
      hoverSynth.triggerAttackRelease(800, 0.1, now);
      hoverSynth.frequency.exponentialRampToValueAtTime(1200, now + 0.1);
    } catch (e) {}
  };

  useEffect(() => {
    const handleInteraction = () => initAudio();
    window.addEventListener('click', handleInteraction, { once: true });
    window.addEventListener('touchstart', handleInteraction, { once: true });
    return () => {
      window.removeEventListener('click', handleInteraction);
      window.removeEventListener('touchstart', handleInteraction);
    };
  }, []);

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setCurrentTimeStr(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setSysMetrics(prev => ({
        cpu: Math.min(98, Math.max(18, prev.cpu + Math.floor(Math.random() * 7 - 3))),
        gpu: Math.min(98, Math.max(25, prev.gpu + Math.floor(Math.random() * 5 - 2))),
        ram: Math.min(95, Math.max(40, prev.ram + (Math.random() > 0.5 ? 1 : -1))),
        fps: isCameraActive ? 58 + Math.floor(Math.random() * 5) : 60,
        temp: isCameraActive ? 44 + Math.floor(Math.random() * 3) : 41
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, [isCameraActive]);

  const getObjectCounts = (objects: string[]) => {
    let human = 0;
    let device = 0;
    let text = 0;
    let vehicle = 0;
    let unknown = 0;

    objects.forEach(obj => {
      const l = obj.toLowerCase();
      if (l === 'person' || l === 'human') human++;
      else if (['cell phone', 'laptop', 'tv', 'monitor', 'keyboard', 'mouse', 'phone', 'tablet', 'remote'].includes(l)) device++;
      else if (['book', 'paper', 'text', 'newspaper', 'card'].includes(l)) text++;
      else if (['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'vehicle', 'train'].includes(l)) vehicle++;
      else unknown++;
    });

    return { human, device, text, vehicle, unknown };
  };

  const calculateAffectiveMetrics = (bs?: { smile: number; frown: number; mouthOpen: number; browRaise: number; eyeBlink: number; pucker: number }) => {
    const smile = bs?.smile ?? consoleState.blendshapes.smile ?? 0;
    const frown = bs?.frown ?? consoleState.blendshapes.frown ?? 0;
    const mouthOpen = bs?.mouthOpen ?? consoleState.blendshapes.mouthOpen ?? 0;
    const browRaise = bs?.browRaise ?? consoleState.blendshapes.browRaise ?? 0;
    const eyeBlink = bs?.eyeBlink ?? consoleState.blendshapes.eyeBlink ?? 0;

    const focus = Math.min(99, Math.max(25, Math.round((1 - eyeBlink) * 65 + browRaise * 25 + 12)));
    const attention = Math.min(99, Math.max(20, Math.round(70 + browRaise * 20 - eyeBlink * 15)));
    const stress = Math.min(99, Math.max(5, Math.round(frown * 75 + (1 - smile) * 15 + 8)));
    const confidence = Math.min(99, Math.max(20, Math.round(smile * 50 + (1 - frown) * 35 + 15)));
    const energy = Math.min(99, Math.max(25, Math.round(smile * 35 + mouthOpen * 40 + 22)));

    return { focus, attention, stress, confidence, energy };
  };

  const GENERAL_HR_QUESTIONS = [
    {
      id: 1,
      title: "Introduction & Career Story",
      question: "Tell me about yourself, your professional background, and what drives your passion in your work.",
      context: "Evaluates self-awareness, communication clarity, and career narrative focus.",
      sampleAnswer: "I am a Senior Software Engineer with over 5 years of experience building scalable web applications and AI-driven user interfaces. In my previous role, I led the development of a real-time data processing pipeline that reduced latency by 40%. I am deeply passionate about creating accessible, intuitive tools that solve real human problems."
    },
    {
      id: 2,
      title: "Strengths & High-Pressure Performance",
      question: "What is your greatest professional strength, and how do you handle tight deadlines or stressful situations?",
      context: "Assesses self-confidence, stress handling, and core competency alignment.",
      sampleAnswer: "My greatest strength is my structured problem-solving approach under tight deadlines. When faced with a high-pressure launch, I prioritize critical path requirements, communicate clearly with cross-functional stakeholders, and break complex tasks into actionable milestones to ensure on-time delivery with zero compromises on quality."
    },
    {
      id: 3,
      title: "Conflict & Problem Resolution",
      question: "Describe a challenging conflict or project setback you experienced in a team, and how you resolved it.",
      context: "Tests emotional intelligence, teamwork, and STAR methodology (Situation, Task, Action, Result).",
      sampleAnswer: "During a major project milestone, our team had conflicting opinions on system architecture. Situation: The backend and frontend teams differed on API design. Task: As technical lead, I needed to align everyone quickly. Action: I facilitated a whiteboarding session where both sides presented data on latency and developer velocity. Result: We agreed on a hybrid architecture that met both performance and speed goals."
    },
    {
      id: 4,
      title: "Career Goals & Vision",
      question: "Where do you see your career evolving in the next 3 to 5 years, and how does this role fit into that vision?",
      context: "Evaluates ambition, long-term commitment, and realistic professional growth alignment.",
      sampleAnswer: "Over the next 3 to 5 years, I aim to deepen my expertise in computer vision and real-time interactive systems while taking on greater technical architecture leadership. This role aligns perfectly with my vision because it allows me to build cutting-edge multimodal applications and mentor junior engineers."
    },
    {
      id: 5,
      title: "Organizational Fit & Value Add",
      question: "Why are you interested in joining our organization, and what unique value will you bring to our team?",
      context: "Measures cultural fit, research effort, and persuasive self-presentation.",
      sampleAnswer: "I have closely followed your organization's innovative work in AI-driven tools. Your focus on developer productivity and user-centric design matches my personal engineering values. I bring a strong technical foundation, proactive communication, and a proven track record of shipping production-ready web platforms."
    }
  ];

  const speakHRQuestion = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.05;
    
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en') && (v.name.includes('Female') || v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Samantha') || v.name.includes('Karen')));
    if (preferredVoice) {
      utterance.voice = preferredVoice;
    }

    utterance.onstart = () => setIsInterviewerSpeaking(true);
    utterance.onend = () => setIsInterviewerSpeaking(false);
    utterance.onerror = () => setIsInterviewerSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  const stopAllMicRecording = () => {
    setIsMicListening(false);
    setMicVolume(0);

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (_) {}
      recognitionRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (_) {}
    }

    if (audioAnimFrameRef.current) {
      cancelAnimationFrame(audioAnimFrameRef.current);
      audioAnimFrameRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(track => track.stop());
      audioStreamRef.current = null;
    }
  };

  const transcribeAudioBlobWithGemini = async (audioBlob: Blob) => {
    if (audioBlob.size < 500) return;
    setIsTranscribing(true);
    setMicStatusText("Transcribing recorded voice with Gemini AI...");

    try {
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      reader.onloadend = async () => {
        const base64Data = (reader.result as string)?.split(',')[1];
        if (!base64Data) {
          setIsTranscribing(false);
          setMicStatusText("");
          return;
        }

        const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY || 'dummy-key' });
        const res = await ai.models.generateContent({
          model: 'gemini-flash-lite-latest',
          contents: [
            {
              inlineData: {
                mimeType: audioBlob.type || 'audio/webm',
                data: base64Data
              }
            },
            {
              text: 'Accurately transcribe this candidate spoken interview response into clear English text. Output ONLY the plain transcription text with no additional intro or explanation.'
            }
          ]
        });

        const transcribedText = res.text?.trim() || "";
        if (transcribedText) {
          setUserAnswerInput(prev => prev ? `${prev.trim()} ${transcribedText}` : transcribedText);
          setMicStatusText("Voice transcribed successfully!");
          setTimeout(() => setMicStatusText(""), 3000);
        } else {
          setMicStatusText("");
        }
        setIsTranscribing(false);
      };
    } catch (e) {
      console.error("Audio transcription error:", e);
      setIsTranscribing(false);
      setMicStatusText("");
    }
  };

  const toggleSpeechRecognition = async () => {
    if (isMicListening) {
      stopAllMicRecording();
      setMicStatusText("Microphone processing...");
      setTimeout(() => {
        if (audioChunksRef.current.length > 0) {
          const combinedBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          if (!userAnswerInput.trim() || userAnswerInput.trim() === baseAnswerTextRef.current.trim()) {
            transcribeAudioBlobWithGemini(combinedBlob);
          } else {
            setMicStatusText("Mic input captured.");
            setTimeout(() => setMicStatusText(""), 2500);
          }
        } else {
          setMicStatusText("");
        }
      }, 300);
      return;
    }

    baseAnswerTextRef.current = userAnswerInput;
    setMicStatusText("Requesting mic access...");

    let audioStream: MediaStream | null = null;
    try {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = audioStream;
    } catch (err: any) {
      console.error("Microphone access error:", err);
      setIsMicListening(false);
      setMicStatusText("Mic access blocked or unavailable. Type your response or click 'Use Sample STAR Answer' below.");
      return;
    }

    setIsMicListening(true);
    playHoverSound();
    setMicStatusText("Listening... Speak clearly into mic");

    // AudioContext + Analyser for real-time live volume level meter
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(audioStream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 128;
      source.connect(analyser);
      audioAnalyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateMicVolume = () => {
        if (audioAnalyserRef.current) {
          audioAnalyserRef.current.getByteFrequencyData(dataArray);
          let sum = 0;
          for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
          }
          const avg = sum / dataArray.length;
          const vol = Math.min(100, Math.round((avg / 120) * 100));
          setMicVolume(vol);
          audioAnimFrameRef.current = requestAnimationFrame(updateMicVolume);
        }
      };
      updateMicVolume();
    } catch (e) {
      console.warn("AudioContext setup notice:", e);
    }

    // MediaRecorder audio capture
    try {
      audioChunksRef.current = [];
      const recorder = new MediaRecorder(audioStream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.start(400);
      mediaRecorderRef.current = recorder;
    } catch (recErr) {
      console.warn("MediaRecorder setup notice:", recErr);
    }

    // Web Speech Recognition
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      try {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
          setIsMicListening(true);
        };

        recognition.onresult = (event: any) => {
          let finalTranscript = '';
          let interimTranscript = '';
          for (let i = 0; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              finalTranscript += event.results[i][0].transcript + ' ';
            } else {
              interimTranscript += event.results[i][0].transcript;
            }
          }
          const spoken = (finalTranscript + interimTranscript).trim();
          if (spoken) {
            const base = baseAnswerTextRef.current;
            setUserAnswerInput(base ? `${base.trim()} ${spoken}` : spoken);
          }
        };

        recognition.onerror = (e: any) => {
          console.warn("Speech recognition notice:", e?.error);
          if (e?.error === 'not-allowed' || e?.error === 'service-not-allowed') {
            setMicStatusText("Speech API restricted - recording voice for Gemini transcription!");
          }
        };

        recognition.onend = () => {
          if (isMicListening && recognitionRef.current) {
            try { recognition.start(); } catch (_) {}
          }
        };

        recognition.start();
        recognitionRef.current = recognition;
      } catch (e) {
        console.error("Speech recognition start error:", e);
      }
    }
  };

  const evaluateInterviewAnswerWithAI = async () => {
    if (isEvaluatingAnswer) return;
    setIsEvaluatingAnswer(true);
    playHoverSound();

    const currentQ = GENERAL_HR_QUESTIONS[interviewStep];
    const currentEmotion = consoleState.emotion || 'neutral';
    const currentMetrics = calculateAffectiveMetrics(consoleState.blendshapes);

    let verbalScore = 85;
    let nonVerbalScore = 80;
    let feedbackText = "";
    let tipsText = "";

    try {
      const apiKey = process.env.GEMINI_API_KEY || process.env.API_KEY;
      if (!apiKey) {
        throw new Error("Gemini API key not configured");
      }
      const ai = new GoogleGenAI({ apiKey });
      const prompt = `You are an executive HR Director evaluating a candidate during a General HR Mock Interview.
Question asked: "${currentQ.question}"
Candidate's response: "${userAnswerInput.trim() || 'Candidate provided a concise answer.'}"
Candidate's real-time detected facial emotion: ${currentEmotion} (Smile factor: ${Math.round((consoleState.blendshapes.smile || 0) * 100)}%, Confidence metric: ${currentMetrics.confidence}%, Stress metric: ${currentMetrics.stress}%).

Provide an evaluation in JSON format with strictly these keys:
{
  "verbalScore": <number between 65 and 98>,
  "nonVerbalScore": <number between 65 and 98>,
  "feedback": "<2 sentence constructive evaluation of response content and facial posture composure>",
  "tips": "<1 actionable recommendation for the next response>"
}`;

      const res = await ai.models.generateContent({
        model: 'gemini-flash-lite-latest',
        contents: prompt,
      });

      const text = res.text?.trim() || "";
      const cleanJson = text.replace(/```json/g, '').replace(/```/g, '').trim();
      const parsed = JSON.parse(cleanJson);
      verbalScore = parsed.verbalScore || 85;
      nonVerbalScore = parsed.nonVerbalScore || 80;
      feedbackText = parsed.feedback || "Articulate response with solid composure and positive emotion telemetry.";
      tipsText = parsed.tips || "Incorporate a concrete metric using the STAR framework.";
    } catch (e: any) {
      console.warn("Gemini evaluation fallback active (API limit/key note):", e?.message || e);
      const answerLen = userAnswerInput.trim().length;
      verbalScore = Math.min(96, Math.max(68, 72 + Math.floor(answerLen / 12)));
      nonVerbalScore = Math.min(98, Math.max(62, currentMetrics.confidence + (currentEmotion === 'happy' ? 12 : 0)));
      
      const isQuotaError = e?.message?.includes('429') || e?.message?.includes('RESOURCE_EXHAUSTED') || e?.status === 429;
      const noteSuffix = isQuotaError ? " (Evaluated via Local Biometric Engine - API Quota Reached)" : "";

      feedbackText = `Your answer addressed the question clearly. Facial telemetry indicates a ${currentEmotion} tone with ${currentMetrics.confidence}% confidence.${noteSuffix}`;
      tipsText = currentMetrics.stress > 30 
        ? "Relax brow tension and maintain steady eye contact with the camera."
        : "Great warmth! Elaborate slightly more on key project outcomes using the STAR framework.";
    }

    const resultItem = {
      question: currentQ.question,
      answer: userAnswerInput.trim() || 'Recorded response with facial expression',
      verbalScore,
      nonVerbalScore,
      emotionAtAnswer: currentEmotion,
      feedback: feedbackText,
      tips: tipsText
    };

    setInterviewHistory(prev => [...prev, resultItem]);
    setInterviewBiometricsLog(prev => [...prev, {
      step: interviewStep + 1,
      emotion: currentEmotion,
      confidence: currentMetrics.confidence,
      stress: currentMetrics.stress,
      positivity: Math.round((consoleState.blendshapes.smile || 0) * 80 + 20)
    }]);

    setIsEvaluatingAnswer(false);
    setUserAnswerInput('');

    if (interviewStep < GENERAL_HR_QUESTIONS.length - 1) {
      setInterviewStep(prev => prev + 1);
    } else {
      setShowFinalReport(true);
    }
  };

  const loadModels = async () => {
    try {
      setStatus('Loading TensorFlow...');
      await tf.ready();

      // Load COCO-SSD with retry
      if (!objectModelRef.current) {
        setStatus('Loading Object Model...');
        let cocoModel = null;
        let cocoAttempts = 0;
        while (cocoAttempts < 3 && !cocoModel) {
          try {
            cocoModel = await cocoSsd.load();
          } catch (e) {
            cocoAttempts++;
            console.warn(`COCO-SSD attempt ${cocoAttempts} failed:`, e);
            if (cocoAttempts < 3) {
              await new Promise(res => setTimeout(res, 1000 * cocoAttempts));
            }
          }
        }
        if (cocoModel) {
          objectModelRef.current = cocoModel;
        }
      }

      // Load MediaPipe FaceLandmarker with CDN mirrors and retries
      if (!faceLandmarkerRef.current) {
        setStatus('Loading Biometric Vision...');
        const wasmSources = [
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm",
          "https://unpkg.com/@mediapipe/tasks-vision@0.10.3/wasm",
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        ];

        const modelSources = [
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          "https://cdn.jsdelivr.net/gh/google-ai-edge/mediapipe-models@main/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        ];

        let loadedLandmarker = null;
        let lastError = null;

        for (const wasmUrl of wasmSources) {
          if (loadedLandmarker) break;
          try {
            const vision = await FilesetResolver.forVisionTasks(wasmUrl);
            for (const modelPath of modelSources) {
              try {
                loadedLandmarker = await FaceLandmarker.createFromOptions(vision, {
                  baseOptions: {
                    modelAssetPath: modelPath,
                  },
                  outputFaceBlendshapes: true,
                  runningMode: "VIDEO",
                  numFaces: 4
                });
                break;
              } catch (mErr) {
                console.warn(`FaceLandmarker create failed with ${modelPath}:`, mErr);
                lastError = mErr;
              }
            }
          } catch (wErr) {
            console.warn(`FilesetResolver failed with ${wasmUrl}:`, wErr);
            lastError = wErr;
          }
        }

        if (loadedLandmarker) {
          faceLandmarkerRef.current = loadedLandmarker;
        } else if (!objectModelRef.current) {
          throw lastError || new Error("Failed to fetch vision model assets from CDN");
        }
      }

      setIsModelLoaded(true);
      setStatus('Idle');
      setErrorMsg(null);
    } catch (err: any) {
      console.error("Failed to load models:", err);
      setStatus('Error loading models');
      setErrorMsg(`Model download failed (${err.message || 'Network fetch error'}). Click RETRY to attempt downloading again.`);
    }
  };

  useEffect(() => {
    loadModels();
    
    return () => {
      stopSession();
    };
  }, []);

  const runDetection = async () => {
    if (!isPlayingRef.current || !videoRef.current || !canvasRef.current || !objectModelRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (video.readyState >= 2 && ctx) {
      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      try {
        const predictions = await objectModelRef.current.detect(video, 10, detectionThresholdRef.current);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const detectedClasses = new Set<string>();

        // --- Smoothing Logic ---
        const newSmoothedBoxes = new Map<string, SmoothedBox>();
        const unassignedPredictions = Array.isArray(predictions) ? [...predictions] : [];

        smoothedBoxesRef.current.forEach((box, id) => {
          let closestIdx = -1;
          let minDist = Infinity;
          unassignedPredictions.forEach((pred, idx) => {
            if (pred && pred.class === box.class && Array.isArray(pred.bbox) && pred.bbox.length >= 4) {
              const [px, py, pw, ph] = pred.bbox;
              const dist = Math.hypot(px + pw/2 - (box.x + box.width/2), py + ph/2 - (box.y + box.height/2));
              if (dist < 150) {
                if (dist < minDist) {
                  minDist = dist;
                  closestIdx = idx;
                }
              }
            }
          });

          if (closestIdx !== -1) {
            const pred = unassignedPredictions[closestIdx];
            if (pred && Array.isArray(pred.bbox) && pred.bbox.length >= 4) {
              const [px, py, pw, ph] = pred.bbox;
              const lerp = 0.15; // Smoothing factor
              box.x += (px - box.x) * lerp;
              box.y += (py - box.y) * lerp;
              box.width += (pw - box.width) * lerp;
              box.height += (ph - box.height) * lerp;
              box.opacity = Math.min(1, box.opacity + 0.1);
              box.score = pred.score;
              
              // Target label position (top right of box)
              const targetLabelX = box.x + box.width + 20;
              const targetLabelY = box.y - 20;
              box.labelX += (targetLabelX - box.labelX) * lerp;
              box.labelY += (targetLabelY - box.labelY) * lerp;

              newSmoothedBoxes.set(id, box);
              detectedClasses.add(box.class);
            }
            unassignedPredictions.splice(closestIdx, 1);
          } else {
            box.opacity -= 0.05; // Fade out
            if (box.opacity > 0) {
              newSmoothedBoxes.set(id, box);
              detectedClasses.add(box.class);
            }
          }
        });

        unassignedPredictions.forEach((pred) => {
          if (!pred || !Array.isArray(pred.bbox) || pred.bbox.length < 4) return;
          const id = Math.random().toString(36).substring(7);
          const [x, y, width, height] = pred.bbox;
          newSmoothedBoxes.set(id, {
            x, y, width, height, class: pred.class, score: pred.score, opacity: 0,
            labelX: x + width + 40, labelY: y - 40
          });
          detectedClasses.add(pred.class);
        });

        smoothedBoxesRef.current = newSmoothedBoxes;

        // --- Drawing Logic ---
        if (showBoundingBoxesRef.current) {
          smoothedBoxesRef.current.forEach((box) => {
          const { x, y, width, height, opacity, labelX, labelY } = box;
          const text = `${box.class} (${Math.round(box.score * 100)}%)`;
          const boxColor = getClassColor(box.class);

          ctx.strokeStyle = boxColor;
          ctx.lineWidth = 2;
          ctx.shadowColor = boxColor;
          ctx.shadowBlur = 8;

          // Draw corners
          const cornerLength = Math.min(18, width / 4, height / 4);
          ctx.beginPath();
          ctx.moveTo(x, y + cornerLength);
          ctx.lineTo(x, y);
          ctx.lineTo(x + cornerLength, y);
          
          ctx.moveTo(x + width - cornerLength, y);
          ctx.lineTo(x + width, y);
          ctx.lineTo(x + width, y + cornerLength);
          
          ctx.moveTo(x + width, y + height - cornerLength);
          ctx.lineTo(x + width, y + height);
          ctx.lineTo(x + width - cornerLength, y + height);
          
          ctx.moveTo(x + cornerLength, y + height);
          ctx.lineTo(x, y + height);
          ctx.lineTo(x, y + height - cornerLength);
          ctx.stroke();

          // Subtle filled overlay with color tint
          ctx.fillStyle = boxColor;
          ctx.globalAlpha = opacity * 0.08;
          ctx.fillRect(x, y, width, height);
          ctx.globalAlpha = 1.0;

          // Crosshair center
          ctx.beginPath();
          ctx.moveTo(x + width / 2 - 5, y + height / 2);
          ctx.lineTo(x + width / 2 + 5, y + height / 2);
          ctx.moveTo(x + width / 2, y + height / 2 - 5);
          ctx.lineTo(x + width / 2, y + height / 2 + 5);
          ctx.strokeStyle = boxColor;
          ctx.lineWidth = 1;
          ctx.stroke();

          // Line to label
          ctx.beginPath();
          ctx.moveTo(x + width, y);
          ctx.lineTo(labelX, labelY + 16);
          ctx.strokeStyle = boxColor;
          ctx.setLineDash([2, 2]);
          ctx.stroke();
          ctx.setLineDash([]);

          // Minimalist Label
          ctx.font = '600 11px "JetBrains Mono", monospace';
          const textWidth = ctx.measureText(text).width;
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(labelX, labelY, textWidth + 10, 18);

          ctx.strokeStyle = boxColor;
          ctx.lineWidth = 1;
          ctx.strokeRect(labelX, labelY, textWidth + 10, 18);

          ctx.fillStyle = boxColor;
          ctx.fillText(text.toUpperCase(), labelX + 5, labelY + 13);
          
          ctx.shadowBlur = 0;
        });
        }

        const classesArray = Array.from(detectedClasses).sort();
        
        let currentEmotion = "neutral";
        let currentBlendshapes = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
        const detectedSubjectsList: BiometricSubject[] = [];
        const faceBoundsList: { minX: number; maxX: number; minY: number; maxY: number }[] = [];
        
        if (faceLandmarkerRef.current) {
          const faceResult = faceLandmarkerRef.current.detectForVideo(video, performance.now());
          const totalFaces = (faceResult && faceResult.faceLandmarks) ? faceResult.faceLandmarks.length : 0;
          
          // 1. Draw Point Cloud for ALL detected faces on secondary canvas
          if (faceCanvasRef.current && totalFaces > 0) {
            const fCanvas = faceCanvasRef.current;
            const fCtx = fCanvas.getContext('2d');
            if (fCtx) {
              fCtx.clearRect(0, 0, fCanvas.width, fCanvas.height);
              const time = performance.now() / 1500;
              const slotWidth = fCanvas.width / totalFaces;

              for (let fIdx = 0; fIdx < totalFaces; fIdx++) {
                const landmarks = faceResult.faceLandmarks?.[fIdx];
                if (!landmarks || !landmarks.length) continue;

                let minX = video.videoWidth, maxX = 0, minY = video.videoHeight, maxY = 0;
                for (const pt of landmarks) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  if (px < minX) minX = px; if (px > maxX) maxX = px;
                  if (py < minY) minY = py; if (py > maxY) maxY = py;
                }
                const faceWidth = Math.max(20, maxX - minX);
                const faceHeight = Math.max(20, maxY - minY);
                const centerX = minX + faceWidth / 2;
                const centerY = minY + faceHeight / 2;

                const scanY = minY + ((Math.sin(time + fIdx * 0.8) + 1) / 2) * faceHeight;
                const slotCenterX = slotWidth * fIdx + slotWidth / 2;
                const scale = Math.min(slotWidth / faceWidth, fCanvas.height / faceHeight) * 0.75;
                const subjColor = fIdx === 0 ? '#00f3ff' : fIdx === 1 ? '#ff007f' : fIdx === 2 ? '#10b981' : '#f59e0b';

                fCtx.shadowColor = subjColor;
                fCtx.shadowBlur = 6;

                for (const pt of landmarks) {
                  const px = pt.x * video.videoWidth;
                  const py = pt.y * video.videoHeight;
                  const dist = Math.abs(py - scanY) / faceHeight;
                  const opacity = Math.max(0.2, 1.0 - dist * 3.5);
                  
                  fCtx.fillStyle = subjColor;
                  fCtx.globalAlpha = opacity;
                  fCtx.beginPath();
                  const drawX = slotCenterX + (px - centerX) * scale;
                  const drawY = fCanvas.height / 2 + (py - centerY) * scale;
                  fCtx.arc(drawX, drawY, 1.8, 0, 2 * Math.PI);
                  fCtx.fill();
                }
                fCtx.globalAlpha = 1.0;
                fCtx.shadowBlur = 0;

                fCtx.fillStyle = subjColor;
                fCtx.font = '600 9px "JetBrains Mono", monospace';
                fCtx.textAlign = 'center';
                fCtx.fillText(`SUBJ 0${fIdx + 1}`, slotCenterX, 14);
                fCtx.textAlign = 'left';

                if (totalFaces > 1 && fIdx > 0) {
                  fCtx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                  fCtx.setLineDash([2, 3]);
                  fCtx.beginPath();
                  fCtx.moveTo(slotWidth * fIdx, 0);
                  fCtx.lineTo(slotWidth * fIdx, fCanvas.height);
                  fCtx.stroke();
                  fCtx.setLineDash([]);
                }
              }
            }
          }

          // 2. Process biometric data & rendering for EACH face on main video canvas
          if (faceResult && faceResult.faceLandmarks && totalFaces > 0) {
            for (let fIdx = 0; fIdx < totalFaces; fIdx++) {
              const landmarks = faceResult.faceLandmarks[fIdx];
              if (!landmarks || !landmarks.length) continue;

              let subjEmotion = 'neutral';
              let subjBlendshapes = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };

              if (faceResult.faceBlendshapes && faceResult.faceBlendshapes[fIdx] && faceResult.faceBlendshapes[fIdx].categories) {
                const blendshapes = faceResult.faceBlendshapes[fIdx].categories;
                const getScore = (name: string) => blendshapes ? (blendshapes.find(b => b.categoryName === name)?.score || 0) : 0;

                subjBlendshapes.smile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
                subjBlendshapes.frown = Math.min(1, (getScore('mouthFrownLeft') + getScore('mouthFrownRight') + getScore('mouthRollLower')) * 5);
                subjBlendshapes.mouthOpen = getScore('jawOpen');
                subjBlendshapes.browRaise = (getScore('browInnerUp') + getScore('browOuterUpLeft') + getScore('browOuterUpRight')) / 3;
                subjBlendshapes.eyeBlink = (getScore('eyeBlinkLeft') + getScore('eyeBlinkRight')) / 2;
                subjBlendshapes.pucker = getScore('mouthPucker');

                const surpriseScore = (getScore('jawOpen') + getScore('browInnerUp')) / 2;
                const angerScore = (getScore('browDownLeft') + getScore('browDownRight') + getScore('mouthPressLeft')) / 3;
                const fearScore = ((getScore('jawOpen') + getScore('browInnerUp') + getScore('mouthStretchLeft') + getScore('mouthStretchRight')) / 4) * 0.6;
                const disgustScore = Math.min(1, (getScore('noseSneerLeft') + getScore('noseSneerRight') + getScore('mouthUpperUpLeft') + getScore('mouthUpperUpRight')) * 4);

                const emotions = [
                  { name: 'happy', score: subjBlendshapes.smile },
                  { name: 'sadness', score: subjBlendshapes.frown },
                  { name: 'surprised', score: surpriseScore },
                  { name: 'angry', score: angerScore },
                  { name: 'fear', score: fearScore },
                  { name: 'disgust', score: disgustScore }
                ];

                const maxE = emotions.reduce((max, e) => e.score > max.score ? e : max, emotions[0]);
                if (maxE.score > 0.2) subjEmotion = maxE.name;
              }

              // Smooth blendshapes per subject
              let smoothedSubjBs = smoothedSubjectBlendshapesRef.current.get(fIdx);
              if (!smoothedSubjBs) {
                smoothedSubjBs = { ...subjBlendshapes };
                smoothedSubjectBlendshapesRef.current.set(fIdx, smoothedSubjBs);
              } else {
                const lerpFactor = 0.25;
                smoothedSubjBs.smile += (subjBlendshapes.smile - smoothedSubjBs.smile) * lerpFactor;
                smoothedSubjBs.frown += (subjBlendshapes.frown - smoothedSubjBs.frown) * lerpFactor;
                smoothedSubjBs.mouthOpen += (subjBlendshapes.mouthOpen - smoothedSubjBs.mouthOpen) * lerpFactor;
                smoothedSubjBs.browRaise += (subjBlendshapes.browRaise - smoothedSubjBs.browRaise) * lerpFactor;
                smoothedSubjBs.eyeBlink += (subjBlendshapes.eyeBlink - smoothedSubjBs.eyeBlink) * lerpFactor;
                smoothedSubjBs.pucker += (subjBlendshapes.pucker - smoothedSubjBs.pucker) * lerpFactor;
              }

              if (fIdx === 0) {
                currentEmotion = subjEmotion;
                currentBlendshapes = { ...smoothedSubjBs };
              }

              let minX = 1, maxX = 0, minY = 1, maxY = 0;
              for (const pt of landmarks) {
                if (pt.x < minX) minX = pt.x;
                if (pt.x > maxX) maxX = pt.x;
                if (pt.y < minY) minY = pt.y;
                if (pt.y > maxY) maxY = pt.y;
              }
              faceBoundsList.push({ minX, maxX, minY, maxY });

              const subjColor = fIdx === 0 ? getEmotionColor(subjEmotion).main : fIdx === 1 ? '#ff007f' : fIdx === 2 ? '#10b981' : '#f59e0b';
              const matchScore = parseFloat((98.2 + (fIdx % 3) * 0.7).toFixed(1));
              const hrVal = 72 + fIdx * 4 + Math.floor(subjBlendshapes.smile * 10);

              detectedSubjectsList.push({
                id: `subj-face-${fIdx}`,
                label: `Subject 0${fIdx + 1}`,
                emotion: subjEmotion,
                matchScore,
                hr: hrVal,
                type: 'face',
                color: subjColor,
                blendshapes: { ...smoothedSubjBs }
              });

              // --- DRAW BIOMETRIC RETICLE OVERLAY FOR FACE fIdx ON MAIN CANVAS ---
              const bx = minX * canvas.width;
              const by = minY * canvas.height;
              const bw = (maxX - minX) * canvas.width;
              const bh = (maxY - minY) * canvas.height;

              ctx.save();
              ctx.strokeStyle = subjColor;
              ctx.lineWidth = 2;
              ctx.shadowColor = subjColor;
              ctx.shadowBlur = 10;

              const corner = Math.min(16, bw / 3, bh / 3);
              const pad = 8;
              const rx = bx - pad;
              const ry = by - pad;
              const rw = bw + pad * 2;
              const rh = bh + pad * 2;

              // Corner brackets
              ctx.beginPath();
              ctx.moveTo(rx, ry + corner); ctx.lineTo(rx, ry); ctx.lineTo(rx + corner, ry);
              ctx.moveTo(rx + rw - corner, ry); ctx.lineTo(rx + rw, ry); ctx.lineTo(rx + rw, ry + corner);
              ctx.moveTo(rx + rw, ry + rh - corner); ctx.lineTo(rx + rw, ry + rh); ctx.lineTo(rx + rw - corner, ry + rh);
              ctx.moveTo(rx + corner, ry + rh); ctx.lineTo(rx, ry + rh); ctx.lineTo(rx, ry + rh - corner);
              ctx.stroke();

              // Subject Header Badge
              ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
              ctx.fillRect(rx, ry - 22, 140, 20);
              ctx.strokeRect(rx, ry - 22, 140, 20);

              ctx.fillStyle = subjColor;
              ctx.font = '700 10px "JetBrains Mono", monospace';
              ctx.fillText(`SUBJECT 0${fIdx + 1} : SCAN`, rx + 6, ry - 8);

              // Telemetry Panel
              const sideX = rx + rw + 8 < canvas.width - 150 ? rx + rw + 8 : rx - 142;
              const sideY = ry;
              ctx.fillStyle = 'rgba(0, 0, 0, 0.82)';
              ctx.fillRect(sideX, sideY, 134, 52);
              ctx.strokeStyle = subjColor;
              ctx.lineWidth = 1;
              ctx.strokeRect(sideX, sideY, 134, 52);

              ctx.fillStyle = '#ffffff';
              ctx.font = '600 9px "JetBrains Mono", monospace';
              ctx.fillText(`EXPR : ${subjEmotion.toUpperCase()}`, sideX + 6, sideY + 14);
              ctx.fillText(`MATCH: ${matchScore}%`, sideX + 6, sideY + 28);
              ctx.fillText(`HR   : ${hrVal} BPM`, sideX + 6, sideY + 42);

              // Animated face scan line
              const scanTime = (performance.now() / 2500) + fIdx * 0.7;
              const scanProgress = (Math.sin(scanTime * Math.PI * 2) + 1) / 2;
              const scanY = ry + scanProgress * rh;

              ctx.strokeStyle = subjColor;
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(rx, scanY);
              ctx.lineTo(rx + rw, scanY);
              ctx.stroke();

              // Illuminated face landmarks near scan line
              ctx.fillStyle = subjColor;
              for (const pt of landmarks) {
                const py = pt.y * canvas.height;
                if (Math.abs(py - scanY) < 14) {
                  ctx.beginPath();
                  ctx.arc(pt.x * canvas.width, py, 1.8, 0, Math.PI * 2);
                  ctx.fill();
                }
              }

              // Feature reticles
              const drawSubjFeature = (ptIdx: number, tag: string) => {
                if (!landmarks[ptIdx]) return;
                const fx = landmarks[ptIdx].x * canvas.width;
                const fy = landmarks[ptIdx].y * canvas.height;
                ctx.fillStyle = subjColor;
                ctx.beginPath();
                ctx.arc(fx, fy, 2.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.font = '600 8px "JetBrains Mono", monospace';
                ctx.fillText(tag, fx + 5, fy + 3);
              };

              drawSubjFeature(33, 'EYE_L');
              drawSubjFeature(263, 'EYE_R');
              drawSubjFeature(1, 'NASAL');
              drawSubjFeature(61, 'ORAL_L');
              drawSubjFeature(291, 'ORAL_R');

              ctx.restore();
            }
          }
        }

        // 3. Fallback: Body Biometric Scan for COCO-SSD person detections without face landmarks
        unassignedPredictions.forEach((pred) => {
          if (pred && pred.class === 'person' && Array.isArray(pred.bbox) && pred.bbox.length >= 4) {
            const [px, py, pw, ph] = pred.bbox;
            const pMinX = px / canvas.width;
            const pMaxX = (px + pw) / canvas.width;
            const pMinY = py / canvas.height;
            const pMaxY = (py + ph) / canvas.height;

            const overlapsWithFace = faceBoundsList.some(fb => {
              return !(pMaxX < fb.minX || pMinX > fb.maxX || pMaxY < fb.minY || pMinY > fb.maxY);
            });

            if (!overlapsWithFace) {
              const bodySubjIdx = detectedSubjectsList.length + 1;
              const subjColor = '#38bdf8';

              detectedSubjectsList.push({
                id: `subj-body-${bodySubjIdx}`,
                label: `Subject 0${bodySubjIdx}`,
                emotion: 'tracking',
                matchScore: 95.4,
                hr: 76,
                type: 'body',
                color: subjColor
              });

              ctx.save();
              ctx.strokeStyle = subjColor;
              ctx.lineWidth = 2;
              ctx.shadowColor = subjColor;
              ctx.shadowBlur = 8;

              const corner = Math.min(20, pw / 4, ph / 4);
              ctx.beginPath();
              ctx.moveTo(px, py + corner); ctx.lineTo(px, py); ctx.lineTo(px + corner, py);
              ctx.moveTo(px + pw - corner, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + corner);
              ctx.moveTo(px + pw, py + ph - corner); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw - corner, py + ph);
              ctx.moveTo(px + corner, py + ph); ctx.lineTo(px, py + ph); ctx.lineTo(px, py + ph - corner);
              ctx.stroke();

              ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
              ctx.fillRect(px, py - 22, 160, 20);
              ctx.strokeRect(px, py - 22, 160, 20);

              ctx.fillStyle = subjColor;
              ctx.font = '700 10px "JetBrains Mono", monospace';
              ctx.fillText(`SUBJECT 0${bodySubjIdx} : BODY TRACKING`, px + 6, py - 8);

              const cx = px + pw / 2;
              const cy = py + ph / 3;
              ctx.beginPath();
              ctx.arc(cx, cy, 12, 0, Math.PI * 2);
              ctx.moveTo(cx - 18, cy); ctx.lineTo(cx + 18, cy);
              ctx.moveTo(cx, cy - 18); ctx.lineTo(cx, cy + 18);
              ctx.stroke();

              ctx.restore();
            }
          }
        });

        // Mirror detection drawing to AI interview modal canvas if open
        if (interviewCanvasRef.current && canvasRef.current) {
          const iCanvas = interviewCanvasRef.current;
          if (iCanvas.width !== canvas.width || iCanvas.height !== canvas.height) {
            iCanvas.width = canvas.width;
            iCanvas.height = canvas.height;
          }
          const iCtx = iCanvas.getContext('2d');
          if (iCtx) {
            iCtx.clearRect(0, 0, iCanvas.width, iCanvas.height);
            iCtx.drawImage(canvas, 0, 0);
          }
        }

        // Throttle React state updates for the console UI to ~10fps
        const now = performance.now();
        if (now - lastStateUpdateTimeRef.current > 100) {
          const smoothingFactor = 0.15;
          const smoothed = smoothedBlendshapesRef.current;
          smoothed.smile += (currentBlendshapes.smile - smoothed.smile) * smoothingFactor;
          smoothed.frown += (currentBlendshapes.frown - smoothed.frown) * smoothingFactor;
          smoothed.mouthOpen += (currentBlendshapes.mouthOpen - smoothed.mouthOpen) * smoothingFactor;
          smoothed.browRaise += (currentBlendshapes.browRaise - smoothed.browRaise) * smoothingFactor;
          smoothed.eyeBlink += (currentBlendshapes.eyeBlink - smoothed.eyeBlink) * smoothingFactor;
          smoothed.pucker += (currentBlendshapes.pucker - smoothed.pucker) * smoothingFactor;

          setConsoleState({
            emotion: currentEmotion,
            objects: classesArray,
            blendshapes: { ...smoothed },
            subjects: detectedSubjectsList
          });
          lastStateUpdateTimeRef.current = now;
        }

        const stateString = `${classesArray.join(',')}|${currentEmotion}`;
        
        if (stateString !== pendingStateRef.current) {
          pendingStateRef.current = stateString;
          
          if (vibeTimeoutRef.current) {
            clearTimeout(vibeTimeoutRef.current);
          }
          
          // Debounce scene updates by 3 seconds
          vibeTimeoutRef.current = setTimeout(async () => {
            if (stateString !== lastStateRef.current) {
              lastStateRef.current = stateString;
              
              const newVibe = await getVibeFromGemini(classesArray, currentEmotion);
              lastPromptRef.current = newVibe;
              setCurrentPrompt(newVibe);
            }
          }, 3000);
        }
      } catch (err) {
        console.error("Detection error:", err);
      }
    }
    
    if (isPlayingRef.current) {
      detectLoopRef.current = requestAnimationFrame(runDetection);
    }
  };

  const startSession = async () => {
    if (!isModelLoaded) return;
    
    try {
      setErrorMsg(null);
      setStatus('Starting camera...');
      
      let stream = streamRef.current;
      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
              width: { ideal: 1280 }, 
              height: { ideal: 720 },
              facingMode: 'user'
            } 
          });
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            await videoRef.current.play().catch(e => console.error("Video play error:", e));
          }
          if (interviewVideoRef.current) {
            interviewVideoRef.current.srcObject = stream;
            await interviewVideoRef.current.play().catch(e => console.error("Interview video play error:", e));
          }
          setIsCameraActive(true);
        } catch (camErr: any) {
          console.error("Camera error:", camErr);
          setStatus('Camera Error');
          setErrorMsg('Camera access denied. Please allow camera access in your browser settings, then refresh the browser page.');
          return;
        }
      }

      // Start detection loop
      if (!isPlayingRef.current) {
        isPlayingRef.current = true;
        detectLoopRef.current = requestAnimationFrame(runDetection);
      }

      setStatus('Vision Active');
      setIsPlaying(true);
      setCurrentPrompt('Biometric Vision Active');

    } catch (err: any) {
      console.error("Setup Error:", err);
      setStatus('Failed to connect');
      setErrorMsg(err.message || 'An unknown error occurred during setup.');
      setInfoMsg(null);
      stopSession(false);
    }
  };

  const stopSession = (closeCamera: boolean = true) => {
    setIsPlaying(false);
    if (vibeTimeoutRef.current) {
      clearTimeout(vibeTimeoutRef.current);
      vibeTimeoutRef.current = null;
    }
    pendingStateRef.current = null;
    
    if (status === 'Connected & Playing' || status === 'Connecting to Lyria API...' || status.includes('Local Synth')) {
      if (!closeCamera && isCameraActive) {
        setStatus('Vision Active');
      } else {
        setStatus('Idle');
      }
    }
    
    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }
    if (sessionRef.current) {
      try { sessionRef.current.conn.close(); } catch (e) {}
      sessionRef.current = null;
    }
    
    setConsoleState({
      emotion: 'neutral',
      objects: [],
      blendshapes: { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 },
      subjects: []
    });
    smoothedBlendshapesRef.current = { smile: 0, frown: 0, mouthOpen: 0, browRaise: 0, eyeBlink: 0, pucker: 0 };
    smoothedBoxesRef.current.clear();
    smoothedSubjectBlendshapesRef.current.clear();
    
    setInfoMsg(null);

    if (closeCamera) {
      isPlayingRef.current = false;
      setCurrentPrompt('Waiting for camera...');
      
      if (detectLoopRef.current) {
        cancelAnimationFrame(detectLoopRef.current);
        detectLoopRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setIsCameraActive(false);
      }
    }
  };

  const activeColorModeObj = CAMERA_COLOR_MODES.find(m => m.id === cameraColorMode) || CAMERA_COLOR_MODES[0];

  return (
    <div className="h-[100dvh] w-full bg-[#03060c] text-white flex flex-col justify-between overflow-hidden font-mono relative p-1.5 sm:p-3 select-none">
      {/* Background Camera & Canvas Elements */}
      <div className="absolute inset-0 z-0">
        {!isCameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-cyan-400/40 z-10 font-mono text-xs tracking-widest">
            <Camera className="w-10 h-10 mb-3 opacity-40 animate-pulse" />
            <p>SYSTEM.VISION_OFFLINE</p>
            <p className="text-[10px] text-white/30 mt-1">PRESS CENTER RETICLE OR START TO ENGAGE</p>
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`absolute inset-0 w-full h-full object-cover transition-all duration-500 ${activeColorModeObj.cssFilter} ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        <canvas
          ref={canvasRef}
          className={`absolute inset-0 w-full h-full object-cover pointer-events-none transition-opacity duration-500 z-[15] ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
        />
        
        {/* Sci-Fi Vignette & Curved Visor Frame Effects */}
        <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,transparent_55%,rgba(2,6,12,0.65)_100%)] z-10" />
        <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(transparent_50%,rgba(0,240,255,0.02)_50%)] bg-[length:100%_4px] z-10" />
        
        {/* Circular Rotating Radar Rings in Background */}
        <div className="absolute inset-0 pointer-events-none z-10 flex items-center justify-center overflow-hidden opacity-25">
          <div className="w-[120vw] h-[120vw] sm:w-[680px] sm:h-[680px] border border-cyan-500/20 rounded-full border-dashed animate-[spin_80s_linear_infinite] shrink-0" />
          <div className="absolute w-[80vw] h-[80vw] sm:w-[480px] sm:h-[480px] border border-cyan-500/10 rounded-full animate-[spin_50s_linear_infinite_reverse] shrink-0" />
          <div className="absolute w-px h-full bg-cyan-500/10" />
          <div className="absolute h-px w-full bg-cyan-500/10" />
        </div>
      </div>

      {/* Visor Glass Border Frame Shell - Pointer-events-none on backdrop frame so center screen camera feed is crystal clear */}
      <div className="relative z-20 w-full h-full border border-cyan-500/30 rounded-[16px] sm:rounded-[28px] overflow-hidden flex flex-col justify-between shadow-[0_0_50px_rgba(0,240,255,0.08)] pointer-events-none">
          
        {/* TOP HEADER BAR */}
        <div className="w-full px-2.5 sm:px-6 py-2 sm:py-2.5 border-b border-cyan-500/30 bg-[#050912]/85 backdrop-blur-md flex items-center justify-between shrink-0 z-30 pointer-events-auto gap-2">
          {/* Left Header Title */}
          <div className="flex items-center gap-2 sm:gap-3 shrink-0">
            <div>
              <div className="flex items-center gap-1.5 sm:gap-2">
                <h1 className="text-xs sm:text-base font-bold tracking-[0.15em] sm:tracking-[0.2em] text-white drop-shadow-[0_0_8px_rgba(0,240,255,0.6)]">
                  NEURAL_VISION
                </h1>
              </div>
              <p className="text-[8px] sm:text-[9px] text-cyan-400/70 font-mono uppercase tracking-widest">
                ENGINE v2.4
              </p>
            </div>
            <div className="hidden md:flex items-center gap-1.5 pl-3 border-l border-white/10 text-[10px] text-white/60">
              <span>STATUS</span>
              <span className="text-cyan-400 font-bold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#00f0ff]" />
                ONLINE
              </span>
            </div>
          </div>

          {/* Center Dynamic Compass Ribbon Bar */}
          <div 
            onClick={() => setCompassHeading(prev => (prev + 45) % 360)}
            title="Click or drag to pivot heading"
            className="flex flex-col items-center justify-center font-mono text-[10px] text-cyan-400/90 tracking-widest cursor-pointer group px-2 sm:px-3 py-1 bg-black/60 border border-cyan-500/30 rounded-lg backdrop-blur-md shadow-[0_0_15px_rgba(0,240,255,0.15)] hover:border-cyan-400 transition-all select-none shrink"
          >
            <div className="flex items-center gap-1.5 sm:gap-2 text-cyan-300 font-bold text-[10px] sm:text-xs">
              <Compass className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-cyan-400 animate-[spin_20s_linear_infinite]" />
              <span className="text-cyan-400 text-[10px] sm:text-xs">▼</span>
              <span className="px-1.5 py-0.5 bg-cyan-950 border border-cyan-400/50 rounded text-[9px] sm:text-[11px] text-cyan-300 shadow-[0_0_8px_#00f0ff]">
                {String(compassHeading).padStart(3, '0')}° {getHeadingCardinal(compassHeading)}
              </span>
            </div>
            
            {/* Dynamic Scale Ribbon */}
            <div className="hidden sm:flex relative w-36 sm:w-56 h-3.5 sm:h-4 overflow-hidden items-center justify-center border-t border-cyan-500/20 pt-0.5">
              <div 
                className="flex items-center gap-3 transition-transform duration-300 ease-out font-mono text-[9px]"
                style={{ transform: `translateX(${-((compassHeading % 360) - 180) * 0.8}px)` }}
              >
                {[
                  { deg: 0, label: 'N' },
                  { deg: 30, label: '30' },
                  { deg: 45, label: 'NE' },
                  { deg: 60, label: '60' },
                  { deg: 90, label: 'E' },
                  { deg: 120, label: '120' },
                  { deg: 135, label: 'SE' },
                  { deg: 150, label: '150' },
                  { deg: 180, label: 'S' },
                  { deg: 210, label: '210' },
                  { deg: 225, label: 'SW' },
                  { deg: 240, label: '240' },
                  { deg: 270, label: 'W' },
                  { deg: 300, label: '300' },
                  { deg: 315, label: 'NW' },
                  { deg: 330, label: '330' },
                  { deg: 360, label: 'N' }
                ].map((tick, idx) => (
                  <span 
                    key={idx} 
                    className={`shrink-0 transition-colors ${
                      Math.abs(tick.deg - (compassHeading % 360)) < 15 
                        ? 'text-cyan-300 font-bold text-[10px] drop-shadow-[0_0_6px_#00f0ff]' 
                        : 'text-white/30 text-[8px]'
                    }`}
                  >
                    {tick.label}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Right Time & Info Controls */}
          <div className="flex items-center gap-2 sm:gap-4 font-mono text-xs shrink-0">
            <div className="text-white/90 font-bold tracking-wider text-[11px] sm:text-xs drop-shadow-[0_0_6px_rgba(255,255,255,0.4)]">
              {currentTimeStr}
            </div>
            <div className="hidden md:flex items-center gap-2 text-cyan-400 text-[10px]">
              <Wifi className="w-3.5 h-3.5" />
              <div className="flex items-center gap-1">
                <BatteryCharging className="w-3.5 h-3.5" />
                <span>87%</span>
              </div>
            </div>
            <button 
              onClick={() => { playHoverSound(); setIsInterviewOpen(true); }}
              onMouseEnter={playHoverSound}
              className="px-2 py-1 sm:px-2.5 sm:py-1 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400/80 rounded-lg text-cyan-300 transition-all flex items-center gap-1 text-[9px] sm:text-[10px] font-bold shadow-[0_0_10px_rgba(0,240,255,0.2)]"
              title="Start AI Mock Interview"
            >
              <Briefcase className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span className="hidden sm:inline">AI INTERVIEW</span>
            </button>
            <button 
              onClick={() => { playHoverSound(); setIsInfoOpen(true); }}
              onMouseEnter={playHoverSound}
              className="p-1 sm:p-1.5 bg-cyan-950/60 hover:bg-cyan-900/80 rounded-full border border-cyan-500/40 text-cyan-300 transition-colors"
              title="App Information"
            >
              <Info className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            </button>
          </div>
        </div>

        {/* MOBILE HUD SECTION SWITCHER TAB BAR */}
        <div className="flex lg:hidden items-center justify-between gap-1 px-3 py-1.5 bg-[#050912]/90 border-b border-cyan-500/20 z-30 pointer-events-auto text-[9px] font-mono">
          <span className="text-white/50 uppercase tracking-wider text-[8px] hidden xs:inline">HUD VIEW:</span>
          <div className="flex items-center gap-1 w-full xs:w-auto justify-around xs:justify-end">
            {[
              { id: 'all', label: 'ALL PANELS' },
              { id: 'affective', label: '🧠 BIOMETRICS' },
              { id: 'reticle', label: '🎯 RETICLE' },
              { id: 'objects', label: '💻 OBJECTS/SYS' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => { playHoverSound(); setMobileHudTab(tab.id as any); }}
                className={`px-2 py-1 rounded text-[9px] font-bold uppercase transition-all ${
                  mobileHudTab === tab.id
                    ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400 shadow-[0_0_8px_#00f0ff]'
                    : 'bg-black/40 text-white/60 border border-white/10 hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* MAIN 3-COLUMN DASHBOARD CONTENT */}
        <div className="relative flex-1 w-full p-2 sm:p-4 overflow-y-auto lg:overflow-hidden flex flex-col lg:flex-row justify-between gap-3 sm:gap-4 pointer-events-none">

          {/* LEFT COLUMN: AFFECTIVE STATE & BIOMETRICS */}
          <div className={`w-full lg:w-72 xl:w-80 flex-col gap-2.5 shrink-0 overflow-y-auto max-h-full pr-1 z-20 pointer-events-auto ${
            mobileHudTab === 'all' || mobileHudTab === 'affective' ? 'flex' : 'hidden lg:flex'
          }`}>
            
            {/* AFFECTIVE STATE PANEL */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-3 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md relative overflow-hidden">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-cyan-400" />
                  AFFECTIVE STATE
                </h3>
              </div>

              {/* ECG Pulse Animation Line */}
              <div className="w-full h-5 mb-1.5 flex items-center justify-center overflow-hidden border-b border-cyan-500/20 pb-0.5">
                <svg className="w-full h-full text-cyan-400 stroke-current fill-none" viewBox="0 0 200 30" preserveAspectRatio="none">
                  <path 
                    d="M0,15 L40,15 L45,5 L50,25 L55,10 L60,15 L100,15 L105,2 L110,28 L115,8 L120,15 L200,15" 
                    strokeWidth="1.5"
                    className="animate-[pulse_1.5s_ease-in-out_infinite]" 
                  />
                </svg>
              </div>

              {/* Active Emotion Display & Multi-subject Tabs */}
              {(() => {
                const faceSubjs = (consoleState.subjects || []).filter(s => s.type === 'face');
                const activeSubj = faceSubjs.find(s => s.id === activeAffectiveTab) || faceSubjs[0];
                const displayEmotion = activeSubj ? activeSubj.emotion : consoleState.emotion;
                const affectiveMetrics = calculateAffectiveMetrics(activeSubj?.blendshapes || consoleState.blendshapes);

                return (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <div className="text-2xl sm:text-3xl font-bold tracking-tight uppercase text-cyan-300 drop-shadow-[0_0_12px_rgba(0,240,255,0.7)]">
                        {displayEmotion}
                      </div>

                      {/* Multi-Subject Tab Selector */}
                      {faceSubjs.length > 1 && (
                        <div className="flex items-center gap-1 bg-black/60 border border-cyan-500/30 p-0.5 rounded">
                          <button
                            onClick={() => { playHoverSound(); setActiveAffectiveTab('all'); }}
                            className={`px-1.5 py-0.5 text-[8px] font-mono uppercase transition-colors ${
                              activeAffectiveTab === 'all' ? 'bg-cyan-500/30 text-cyan-300 font-bold border border-cyan-400' : 'text-white/60 hover:text-white'
                            }`}
                          >
                            ALL
                          </button>
                          {faceSubjs.map((s, idx) => (
                            <button
                              key={s.id}
                              onClick={() => { playHoverSound(); setActiveAffectiveTab(s.id); }}
                              className={`px-1.5 py-0.5 text-[8px] font-mono uppercase transition-colors flex items-center gap-1 ${
                                activeAffectiveTab === s.id ? 'bg-cyan-500/30 text-cyan-300 font-bold border border-cyan-400' : 'text-white/60 hover:text-white'
                              }`}
                            >
                              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: s.color }} />
                              SUBJ 0{idx + 1}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>

                    {/* Metric Bars & Live Line Graphs */}
                    <div className="space-y-1.5">
                      {[
                        { label: 'FOCUS', value: affectiveMetrics.focus, color: '#00f0ff' },
                        { label: 'ATTENTION', value: affectiveMetrics.attention, color: '#38bdf8' },
                        { label: 'STRESS', value: affectiveMetrics.stress, color: '#f43f5e' },
                        { label: 'CONFIDENCE', value: affectiveMetrics.confidence, color: '#10b981' },
                        { label: 'ENERGY', value: affectiveMetrics.energy, color: '#f59e0b' },
                      ].map((item) => (
                        <div key={item.label} className="bg-black/40 border border-cyan-500/20 p-1.5 rounded-md">
                          <div className="flex items-center justify-between text-[9px] mb-1 font-mono">
                            <span className="text-white/70 tracking-wider flex items-center gap-1">
                              <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                              {item.label}
                            </span>
                            <div className="flex items-center gap-2">
                              <MetricSparkline data={metricHistory[item.label] || [50, item.value]} color={item.color} />
                              <span className="font-bold text-cyan-300 w-7 text-right">{item.value}%</span>
                            </div>
                          </div>
                          <div className="h-1 bg-cyan-950 border border-cyan-500/30 overflow-hidden rounded-sm">
                            <motion.div 
                              className="h-full shadow-[0_0_8px_#00f0ff]"
                              style={{ backgroundColor: item.color }}
                              initial={{ width: 0 }}
                              animate={{ width: `${item.value}%` }}
                              transition={{ type: 'spring', bounce: 0, duration: 0.4 }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* BIOMETRIC PANEL */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-3 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md">
              <h3 className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                BIOMETRIC
              </h3>

              {(() => {
                const faceSubjs = (consoleState.subjects || []).filter(s => s.type === 'face');
                const activeSubj = faceSubjs.find(s => s.id === activeAffectiveTab) || faceSubjs[0];
                const hr = activeSubj ? activeSubj.hr : 72;
                const emotion = activeSubj ? activeSubj.emotion : consoleState.emotion;

                return (
                  <div className="space-y-1.5 text-[10px] font-mono">
                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/70">
                        <Heart className="w-3.5 h-3.5 text-rose-400 animate-pulse" />
                        <span>HEART RATE</span>
                      </div>
                      <span className="font-bold text-cyan-300">{hr} BPM</span>
                    </div>

                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/70">
                        <Eye className="w-3.5 h-3.5 text-cyan-400" />
                        <span>EYE TRACKING</span>
                      </div>
                      <span className="font-bold text-emerald-400 uppercase">ACTIVE</span>
                    </div>

                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/70">
                        <ScanFace className="w-3.5 h-3.5 text-amber-400" />
                        <span>FACIAL STATUS</span>
                      </div>
                      <span className="font-bold text-cyan-300 uppercase">{emotion}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-white/70">
                        <Thermometer className="w-3.5 h-3.5 text-cyan-400" />
                        <span>BODY TEMP.</span>
                      </div>
                      <span className="font-bold text-cyan-300">36.6 °C</span>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* SECONDARY LANDMARK SCANNER CANVAS */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-2 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md flex flex-col h-28 relative overflow-hidden">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[9px] font-bold text-cyan-400/80 uppercase tracking-widest flex items-center gap-1">
                  <ScanFace className="w-3 h-3" /> LANDMARK MESH
                </span>
                {(consoleState.subjects || []).length > 0 && (
                  <span className="text-[8px] font-bold text-cyan-300 bg-cyan-950 px-1.5 py-0.5 border border-cyan-500/40">
                    TARGETS: {(consoleState.subjects || []).length}
                  </span>
                )}
              </div>
              <div className="relative flex-1 w-full border border-cyan-500/20 bg-black/50 overflow-hidden flex items-center justify-center">
                <canvas
                  ref={faceCanvasRef}
                  width={280}
                  height={140}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${isCameraActive ? 'opacity-100' : 'opacity-0'}`}
                />
              </div>
            </div>
          </div>

          {/* CENTER COLUMN: MAIN RETICLE & HUD TARGETING FRAME */}
          <div className={`flex-1 relative flex-col items-center justify-center min-h-[220px] sm:min-h-[250px] pointer-events-none z-10 ${
            mobileHudTab === 'all' || mobileHudTab === 'reticle' ? 'flex' : 'hidden lg:flex'
          }`}>
            {/* Futuristic Center Targeting Reticle Bracket Corners */}
            <div className="relative w-48 h-48 sm:w-72 sm:h-72 border border-cyan-500/20 flex items-center justify-center">
              {/* Top-Left Corner Bracket */}
              <div className="absolute -top-1 -left-1 w-6 h-6 sm:w-8 sm:h-8 border-t-2 border-l-2 border-cyan-400" />
              {/* Top-Right Corner Bracket */}
              <div className="absolute -top-1 -right-1 w-6 h-6 sm:w-8 sm:h-8 border-t-2 border-r-2 border-cyan-400" />
              {/* Bottom-Left Corner Bracket */}
              <div className="absolute -bottom-1 -left-1 w-6 h-6 sm:w-8 sm:h-8 border-b-2 border-l-2 border-cyan-400" />
              {/* Bottom-Right Corner Bracket */}
              <div className="absolute -bottom-1 -right-1 w-6 h-6 sm:w-8 sm:h-8 border-b-2 border-r-2 border-cyan-400" />

              {/* Center HUD Badge Overlay */}
              <div className="absolute bottom-3 sm:bottom-4 flex flex-col items-center gap-1 bg-black/60 backdrop-blur-md border border-cyan-500/40 px-2.5 sm:px-3 py-1 rounded text-center">
                <span className="text-[8px] sm:text-[9px] font-bold font-mono tracking-widest text-cyan-300">
                  {isCameraActive ? 'VISION_SYNC ONLINE' : 'SYSTEM STANDBY'}
                </span>
                {/* Audio Wave Form Pulse */}
                <div className="flex items-center gap-0.5 h-2.5">
                  {[40, 70, 100, 60, 90, 50, 80, 40].map((h, i) => (
                    <div 
                      key={i} 
                      className="w-0.5 bg-cyan-400 rounded-full animate-pulse" 
                      style={{ height: `${isCameraActive ? h : 20}%`, animationDelay: `${i * 0.15}s` }} 
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: OBJECT DETECTION, SYSTEM INFO & AI STATUS */}
          <div className={`w-full lg:w-72 xl:w-80 flex-col gap-2.5 shrink-0 overflow-y-auto max-h-full pl-0 lg:pl-1 z-20 pointer-events-auto ${
            mobileHudTab === 'all' || mobileHudTab === 'objects' ? 'flex' : 'hidden lg:flex'
          }`}>
            
            {/* OBJECT DETECTION PANEL */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-3 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md">
              <h3 className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                OBJECT DETECTION
              </h3>

              {(() => {
                const counts = getObjectCounts(consoleState.objects || []);
                return (
                  <div className="space-y-1.5 text-[10px] font-mono">
                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/80">
                        <UserCheck className="w-3.5 h-3.5 text-cyan-400" />
                        <span>HUMAN</span>
                      </div>
                      <span className="font-bold text-cyan-300">{counts.human}</span>
                    </div>

                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/80">
                        <Smartphone className="w-3.5 h-3.5 text-cyan-400" />
                        <span>DEVICE</span>
                      </div>
                      <span className="font-bold text-cyan-300">{counts.device}</span>
                    </div>

                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/80">
                        <FileText className="w-3.5 h-3.5 text-cyan-400" />
                        <span>TEXT</span>
                      </div>
                      <span className="font-bold text-cyan-300">{counts.text}</span>
                    </div>

                    <div className="flex items-center justify-between border-b border-cyan-500/15 pb-1">
                      <div className="flex items-center gap-2 text-white/80">
                        <Car className="w-3.5 h-3.5 text-cyan-400" />
                        <span>VEHICLE</span>
                      </div>
                      <span className="font-bold text-cyan-300">{counts.vehicle}</span>
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-white/80">
                        <HelpCircle className="w-3.5 h-3.5 text-cyan-400/60" />
                        <span>UNKNOWN</span>
                      </div>
                      <span className="font-bold text-cyan-300">{counts.unknown}</span>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* SYSTEM INFO PANEL */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-3 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md">
              <h3 className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <Monitor className="w-3.5 h-3.5 text-cyan-400" />
                SYSTEM INFO
              </h3>

              <div className="space-y-1.5 text-[9px] font-mono">
                <div>
                  <div className="flex justify-between mb-0.5">
                    <span className="text-white/60">CPU</span>
                    <span className="font-bold text-cyan-300">{sysMetrics.cpu}%</span>
                  </div>
                  <div className="h-1 bg-cyan-950/80 border border-cyan-500/30 overflow-hidden rounded-sm">
                    <div className="h-full bg-cyan-400 shadow-[0_0_6px_#00f0ff]" style={{ width: `${sysMetrics.cpu}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-0.5">
                    <span className="text-white/60">GPU</span>
                    <span className="font-bold text-cyan-300">{sysMetrics.gpu}%</span>
                  </div>
                  <div className="h-1 bg-cyan-950/80 border border-cyan-500/30 overflow-hidden rounded-sm">
                    <div className="h-full bg-cyan-400 shadow-[0_0_6px_#00f0ff]" style={{ width: `${sysMetrics.gpu}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-0.5">
                    <span className="text-white/60">RAM</span>
                    <span className="font-bold text-cyan-300">{sysMetrics.ram}%</span>
                  </div>
                  <div className="h-1 bg-cyan-950/80 border border-cyan-500/30 overflow-hidden rounded-sm">
                    <div className="h-full bg-cyan-400 shadow-[0_0_6px_#00f0ff]" style={{ width: `${sysMetrics.ram}%` }} />
                  </div>
                </div>

                <div className="flex justify-between pt-1 border-t border-cyan-500/15">
                  <span className="text-white/60">FPS</span>
                  <span className="font-bold text-cyan-300">{sysMetrics.fps} FPS</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-white/60">TEMP</span>
                  <span className="font-bold text-cyan-300">{sysMetrics.temp} °C</span>
                </div>
              </div>
            </div>

            {/* AI STATUS PANEL */}
            <div className="bg-[#080d19]/90 border border-cyan-500/30 p-3 rounded-lg shadow-[0_0_20px_rgba(0,0,0,0.6)] backdrop-blur-md">
              <h3 className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest mb-2 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-cyan-400" />
                AI STATUS
              </h3>

              <div className="space-y-1 text-[10px] font-mono">
                <div className="flex justify-between">
                  <span className="text-white/60">VISION_ENGINE</span>
                  <span className="font-bold text-white/90">v2.4</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">NEURAL LINK</span>
                  <span className="font-bold text-cyan-400">CONNECTED</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/60">LEARNING</span>
                  <span className="font-bold text-cyan-400 flex items-center gap-1">
                    ACTIVE
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  </span>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* BOTTOM CONTROL VISOR DOCK BAR */}
        <div className="w-full px-4 py-2 sm:py-2.5 border-t border-cyan-500/20 bg-black/60 backdrop-blur-lg flex flex-col items-center gap-1 shrink-0 z-30 pointer-events-auto">
          
          {/* Floating Controls Row */}
          <div className="flex items-center justify-center gap-2 sm:gap-4 max-w-2xl w-full overflow-x-auto no-scrollbar py-0.5 px-2">
            
            {/* CAMERA BUTTON */}
            <button
              onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all ${
                isCameraActive 
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.3)]' 
                  : 'bg-white/5 text-white/60 border-white/20 hover:text-white'
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>CAMERA</span>
            </button>

            {/* RECORD BUTTON */}
            <button
              onClick={() => { playHoverSound(); setIsRecording(!isRecording); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all ${
                isRecording 
                  ? 'bg-red-500/20 text-red-400 border-red-500 shadow-[0_0_12px_rgba(239,68,68,0.4)]' 
                  : 'bg-white/5 text-white/60 border-white/20 hover:text-white'
              }`}
            >
              <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-ping' : 'bg-red-500/70'}`} />
              <span>RECORD</span>
            </button>

            {/* SCAN BUTTON */}
            <button
              onClick={() => { playHoverSound(); setIsScanActive(!isScanActive); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all ${
                isScanActive 
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.3)]' 
                  : 'bg-white/5 text-white/60 border-white/20 hover:text-white'
              }`}
            >
              <ScanFace className="w-3.5 h-3.5" />
              <span>SCAN</span>
            </button>

            {/* CENTER RETICLE ACTION TRIGGER BUTTON */}
            <button
              onClick={() => { playHoverSound(); isCameraActive ? stopSession(true) : startSession(); }}
              onMouseEnter={playHoverSound}
              disabled={!isModelLoaded}
              className="relative group p-1 mx-1 sm:mx-3 transition-transform active:scale-95 shrink-0"
              title={isCameraActive ? "Stop System" : "Engage System"}
            >
              {/* Outer Glowing Ring */}
              <div className={`absolute inset-0 rounded-full border-2 border-cyan-400 transition-all ${isCameraActive ? 'animate-ping opacity-30 bg-cyan-400/20' : 'group-hover:scale-110'}`} />
              <div className="w-10 h-10 sm:w-11 sm:h-11 rounded-full bg-cyan-950 border-2 border-cyan-400 flex items-center justify-center text-cyan-300 shadow-[0_0_20px_#00f0ff]">
                <Target className="w-5 h-5 animate-[spin_10s_linear_infinite]" />
              </div>
            </button>

            {/* VOICE BUTTON */}
            <button
              onClick={() => { playHoverSound(); setIsVoiceActive(!isVoiceActive); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all ${
                isVoiceActive 
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.3)]' 
                  : 'bg-white/5 text-white/60 border-white/20 hover:text-white'
              }`}
            >
              <Mic className="w-3.5 h-3.5" />
              <span>VOICE</span>
            </button>


            {/* SETTINGS BUTTON */}
            <button
              onClick={() => { playHoverSound(); setIsSettingsOpen(!isSettingsOpen); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all ${
                isSettingsOpen 
                  ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.3)]' 
                  : 'bg-white/5 text-white/60 border-white/20 hover:text-white'
              }`}
            >
              <Settings className="w-3.5 h-3.5" />
              <span>SETTINGS</span>
            </button>

            {/* AI MOCK INTERVIEW BUTTON */}
            <button
              onClick={() => { playHoverSound(); setIsInterviewOpen(true); }}
              onMouseEnter={playHoverSound}
              className={`flex flex-col items-center gap-0.5 px-3 py-1 rounded border text-[9px] font-mono font-bold uppercase transition-all bg-gradient-to-r from-cyan-950 to-blue-950 text-cyan-300 border-cyan-400 shadow-[0_0_12px_rgba(0,240,255,0.3)] hover:brightness-125`}
            >
              <Briefcase className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span>INTERVIEW</span>
            </button>

          </div>

          {/* Bottom Sub-Label Bar */}
          <div className="w-full flex items-center justify-between text-[8px] sm:text-[9px] font-mono text-cyan-400/60 px-2 sm:px-6">
            <div className="flex items-center gap-1">
              <span>NEURAL INTERFACE</span>
              <span className="text-cyan-400">●●●●●●●●</span>
            </div>

            <div className="flex items-center gap-1 text-cyan-400/80 font-bold">
              <span>SYSTEM_READY :: ONLINE</span>
            </div>
          </div>

        </div>
      </div>

      {/* Error Modal */}
      <AnimatePresence>
        {errorMsg && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-zinc-900 border border-red-500/50 p-6 max-w-md w-full shadow-[0_0_40px_rgba(239,68,68,0.2)] relative"
            >
              <div className="flex items-start gap-4 mb-6">
                <div className="p-3 bg-red-500/10 border border-red-500/30 shrink-0">
                  <AlertCircle className="w-6 h-6 text-red-500" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-red-500 uppercase tracking-widest">{status}</h3>
                  <p className="text-sm mt-2 text-red-400/80 leading-relaxed">{errorMsg}</p>
                </div>
              </div>
              
              <div className="flex gap-3">
                {status === 'Error loading models' && (
                  <button 
                    onClick={() => {
                      playHoverSound();
                      setErrorMsg(null);
                      loadModels();
                    }}
                    className="flex-1 py-3 bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400 text-xs font-mono font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Retry Models
                  </button>
                )}
                {(errorMsg.toLowerCase().includes('lyria') || errorMsg.toLowerCase().includes('service') || status === 'Failed to connect') && (
                  <button 
                    onClick={() => {
                      playHoverSound();
                      setErrorMsg(null);
                      startSession();
                    }}
                    className="flex-1 py-3 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/50 text-cyan-300 text-xs font-mono font-bold uppercase tracking-widest transition-colors flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Retry Audio
                  </button>
                )}
                <button 
                  onClick={() => setErrorMsg(null)}
                  className={`flex-1 py-3 text-xs font-mono font-bold uppercase tracking-widest transition-colors ${
                    status === 'Camera Error' 
                      ? 'bg-red-500/20 hover:bg-red-500/30 border border-red-500/50 text-red-400' 
                      : 'bg-white/5 hover:bg-white/10 border border-white/20 text-white/70'
                  }`}
                >
                  Dismiss
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Info Toast (Center Bottom) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex flex-col justify-end items-center pointer-events-none z-30 w-[calc(100%-2rem)] sm:w-full max-w-md">
        <AnimatePresence>
          {infoMsg && !errorMsg && (
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="bg-black/80 backdrop-blur-md border border-white/30 p-4 flex items-start gap-3 text-white shadow-[0_0_20px_rgba(255,255,255,0.1)] mb-4 w-full"
            >
              <Music className="w-5 h-5 shrink-0 mt-0.5" />
              <div>
                <h3 className="font-bold text-sm">{status}</h3>
                <p className="text-xs mt-1 text-white/80">{infoMsg}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Info Modal */}
      <AnimatePresence>
        {isInfoOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm pointer-events-auto"
            onClick={() => setIsInfoOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-white/20 p-6 max-w-lg w-full shadow-[0_0_40px_rgba(0,0,0,0.8)] relative max-h-[90vh] overflow-y-auto"
            >
              <div className="flex flex-col-reverse sm:flex-row sm:items-start justify-between gap-4 mb-4">
                <h2 className="text-xl font-bold text-white flex items-center gap-2 self-start">
                  <Info className="w-5 h-5 shrink-0" />
                  About Vision to Music
                </h2>
                <button 
                  onClick={() => setIsInfoOpen(false)}
                  className="p-2 shrink-0 border border-white/20 bg-black/50 hover:bg-white/10 text-white/50 hover:text-white transition-colors self-end sm:self-auto"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="space-y-4 text-sm text-white/80 leading-relaxed">
                <p>
                  <strong>Vision to Music</strong> uses your device's camera to analyze your facial expressions and the objects around you in real-time.
                </p>
                <p>
                  Based on this visual data, it generates a continuous, procedural ambient soundscape that matches your mood and environment.
                </p>
                <ul className="list-disc pl-5 space-y-2 text-white/70">
                  <li><strong>Biometric Scan:</strong> Tracks your facial landmarks to determine your current emotion (happy, sad, surprised, angry, fear, disgust).</li>
                  <li><strong>Entities:</strong> Detects objects in your environment (like laptops, cups, plants) to influence the system experience.</li>
                </ul>
                <p className="text-xs text-white/50 mt-4 pt-4 border-t border-white/10">
                  Note: All processing happens locally in your browser or via secure API calls. No video data is saved or transmitted.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Modal */}
      <AnimatePresence>
        {isSettingsOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 bg-black/80 backdrop-blur-md pointer-events-auto"
            onClick={() => setIsSettingsOpen(false)}
          >
            <motion.div 
              initial={{ scale: 0.95, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 10 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-[#080d1a] border-2 border-cyan-500/50 rounded-2xl p-5 sm:p-7 max-w-2xl w-full shadow-[0_0_60px_rgba(0,240,255,0.25)] relative max-h-[90vh] overflow-y-auto font-mono text-white text-xs"
            >
              {/* Header */}
              <div className="flex items-center justify-between pb-4 border-b border-cyan-500/30 mb-5">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-cyan-950 border border-cyan-400 rounded-lg text-cyan-300 shadow-[0_0_12px_#00f0ff]">
                    <Settings className="w-5 h-5 animate-[spin_12s_linear_infinite]" />
                  </div>
                  <div>
                    <h2 className="text-base sm:text-lg font-bold tracking-widest text-white flex items-center gap-2">
                      SYSTEM_CONFIG :: SETTINGS
                    </h2>
                    <p className="text-[10px] text-cyan-400/70 font-mono">VISION_SYNC HARDWARE & INTERFACE CONTROL</p>
                  </div>
                </div>
                <button 
                  onClick={() => setIsSettingsOpen(false)}
                  className="p-2 border border-cyan-500/30 bg-cyan-950/50 hover:bg-cyan-900/80 text-cyan-300 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-6">
                {/* CAMERA & COLOR PRESETS */}
                <div className="bg-black/50 border border-cyan-500/20 p-4 rounded-xl space-y-3">
                  <div className="flex items-center justify-between text-cyan-300 font-bold uppercase tracking-wider text-xs border-b border-cyan-500/20 pb-2">
                    <span className="flex items-center gap-2"><Camera className="w-4 h-4 text-cyan-400" /> CAMERA COLOR PRESET</span>
                    <span className="text-[10px] text-cyan-400/80">{activeColorModeObj.name}</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                    {CAMERA_COLOR_MODES.map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() => { playHoverSound(); setCameraColorMode(mode.id); }}
                        className={`p-2 rounded border text-left flex flex-col gap-1 transition-all ${
                          cameraColorMode === mode.id
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-200 shadow-[0_0_12px_rgba(0,240,255,0.3)]'
                            : 'bg-white/5 border-white/10 hover:border-cyan-500/50 text-white/70 hover:text-white'
                        }`}
                      >
                        <span className="font-bold text-[11px]">{mode.name}</span>
                        <span className="text-[9px] opacity-60 leading-tight">{mode.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* DETECTION & HUD OVERLAYS */}
                <div className="bg-black/50 border border-cyan-500/20 p-4 rounded-xl space-y-4">
                  <div className="flex items-center justify-between text-cyan-300 font-bold uppercase tracking-wider text-xs border-b border-cyan-500/20 pb-2">
                    <span className="flex items-center gap-2"><Target className="w-4 h-4 text-cyan-400" /> VISION & DETECTION CONTROLS</span>
                  </div>
                  
                  {/* Detection Sensitivity Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-white/80">
                      <span>Detection Confidence Threshold</span>
                      <span className="text-cyan-400 font-bold">{Math.round(detectionThreshold * 100)}%</span>
                    </div>
                    <input 
                      type="range"
                      min="0.2"
                      max="0.8"
                      step="0.05"
                      value={detectionThreshold}
                      onChange={(e) => setDetectionThreshold(parseFloat(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-cyan-950 rounded"
                    />
                  </div>

                  {/* Toggles Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                    <button
                      onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                      className={`p-2.5 rounded border flex items-center justify-between transition-all ${
                        showBoundingBoxes ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300' : 'bg-white/5 border-white/10 text-white/50'
                      }`}
                    >
                      <span className="text-[11px] font-bold">Bounding Boxes</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-cyan-500/30">
                        {showBoundingBoxes ? 'ON' : 'OFF'}
                      </span>
                    </button>

                    <button
                      onClick={() => setShowFaceMesh(!showFaceMesh)}
                      className={`p-2.5 rounded border flex items-center justify-between transition-all ${
                        showFaceMesh ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300' : 'bg-white/5 border-white/10 text-white/50'
                      }`}
                    >
                      <span className="text-[11px] font-bold">Face Mesh</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-cyan-500/30">
                        {showFaceMesh ? 'ON' : 'OFF'}
                      </span>
                    </button>

                    <button
                      onClick={() => setShowBiometrics(!showBiometrics)}
                      className={`p-2.5 rounded border flex items-center justify-between transition-all ${
                        showBiometrics ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300' : 'bg-white/5 border-white/10 text-white/50'
                      }`}
                    >
                      <span className="text-[11px] font-bold">Biometrics HUD</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-cyan-500/30">
                        {showBiometrics ? 'ON' : 'OFF'}
                      </span>
                    </button>
                  </div>
                </div>

                {/* AUDIO & SONIFICATION ENGINE */}
                <div className="bg-black/50 border border-cyan-500/20 p-4 rounded-xl space-y-4">
                  <div className="flex items-center justify-between text-cyan-300 font-bold uppercase tracking-wider text-xs border-b border-cyan-500/20 pb-2">
                    <span className="flex items-center gap-2"><Volume2 className="w-4 h-4 text-cyan-400" /> AUDIO SONIFICATION ENGINE</span>
                  </div>

                  {/* Master Volume Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-white/80">
                      <span>Master Volume</span>
                      <span className="text-cyan-400 font-bold">{masterVolume}%</span>
                    </div>
                    <input 
                      type="range"
                      min="0"
                      max="100"
                      value={masterVolume}
                      onChange={(e) => setMasterVolume(parseInt(e.target.value))}
                      className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-cyan-950 rounded"
                    />
                  </div>

                  {/* Music Vibe Selector */}
                  <div className="space-y-1.5">
                    <span className="text-[11px] text-white/80">Sonification Style Preset</span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {[
                        { id: 'ambient', name: 'Ambient Space' },
                        { id: 'cyberpunk', name: 'Cyber Synth' },
                        { id: 'zen', name: 'Zen Minimal' },
                        { id: 'glitch', name: 'Industrial' }
                      ].map((vibe) => (
                        <button
                          key={vibe.id}
                          onClick={() => setMusicVibeStyle(vibe.id)}
                          className={`p-2 rounded border text-center text-[10px] font-bold transition-all ${
                            musicVibeStyle === vibe.id
                              ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_10px_rgba(0,240,255,0.2)]'
                              : 'bg-white/5 border-white/10 text-white/60 hover:text-white'
                          }`}
                        >
                          {vibe.name}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* COMPASS & HARDWARE */}
                <div className="bg-black/50 border border-cyan-500/20 p-4 rounded-xl space-y-3">
                  <div className="flex items-center justify-between text-cyan-300 font-bold uppercase tracking-wider text-xs border-b border-cyan-500/20 pb-2">
                    <span className="flex items-center gap-2"><Compass className="w-4 h-4 text-cyan-400" /> COMPASS & GYROSCOPE</span>
                    <span className="text-cyan-400 font-bold">{compassHeading}° {getHeadingCardinal(compassHeading)}</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <button
                      onClick={() => setIsGyroActive(!isGyroActive)}
                      className={`p-2.5 rounded border flex items-center justify-between transition-all ${
                        isGyroActive ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300' : 'bg-white/5 border-white/10 text-white/50'
                      }`}
                    >
                      <span className="text-[11px] font-bold">Gyro Heading Auto-Sync</span>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-cyan-500/30">
                        {isGyroActive ? 'ACTIVE' : 'STATIC'}
                      </span>
                    </button>

                    <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded p-2">
                      <span className="text-[10px] text-white/60 shrink-0">Pivot:</span>
                      <button 
                        onClick={() => setCompassHeading(prev => (prev - 15 + 360) % 360)}
                        className="flex-1 py-1 bg-cyan-950 border border-cyan-500/40 hover:bg-cyan-900 text-cyan-300 rounded font-bold text-[10px]"
                      >
                        ◄ -15°
                      </button>
                      <button 
                        onClick={() => setCompassHeading(prev => (prev + 15) % 360)}
                        className="flex-1 py-1 bg-cyan-950 border border-cyan-500/40 hover:bg-cyan-900 text-cyan-300 rounded font-bold text-[10px]"
                      >
                        +15° ►
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Footer Actions */}
              <div className="mt-6 pt-4 border-t border-cyan-500/30 flex items-center justify-between gap-3">
                <button
                  onClick={() => {
                    setShowFaceMesh(true);
                    setShowBoundingBoxes(true);
                    setShowBiometrics(true);
                    setDetectionThreshold(0.5);
                    setMasterVolume(80);
                    setMusicVibeStyle('ambient');
                    setCameraColorMode('normal');
                    setCompassHeading(45);
                    setIsGyroActive(true);
                  }}
                  className="px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/20 text-white/70 hover:text-white rounded-lg text-xs font-bold uppercase transition-colors"
                >
                  Reset Defaults
                </button>

                <button
                  onClick={() => setIsSettingsOpen(false)}
                  className="px-6 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400 text-cyan-300 rounded-lg text-xs font-bold uppercase tracking-wider shadow-[0_0_15px_rgba(0,240,255,0.3)] transition-all"
                >
                  Apply & Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* AI Mock Interview Full-Screen Page Modal */}
      <AnimatePresence>
        {isInterviewOpen && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            className="fixed inset-0 z-50 bg-[#040812]/95 backdrop-blur-2xl flex flex-col font-mono text-white pointer-events-auto overflow-hidden"
          >
            {/* Top Navigation Bar */}
            <div className="w-full px-4 sm:px-6 py-3 border-b border-cyan-500/30 bg-[#080d1a] flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-cyan-950 border border-cyan-400 rounded-lg text-cyan-300 shadow-[0_0_15px_#00f0ff]">
                  <Briefcase className="w-5 h-5 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-sm sm:text-base font-bold tracking-widest text-white flex items-center gap-2">
                    AI MOCK INTERVIEW <span className="text-xs px-2 py-0.5 bg-cyan-500/20 border border-cyan-400 text-cyan-300 rounded">GENERAL HR MODE</span>
                  </h2>
                  <p className="text-[10px] text-cyan-400/70 font-mono">FACIAL EMOTION & RESPONSE COMPOSURE ANALYZER</p>
                </div>
              </div>

              {/* Step indicator */}
              <div className="hidden sm:flex items-center gap-2 bg-black/50 border border-cyan-500/30 px-3 py-1.5 rounded-lg text-xs font-bold text-cyan-300">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>QUESTION {interviewStep + 1} OF {GENERAL_HR_QUESTIONS.length}</span>
              </div>

              <button 
                onClick={() => { playHoverSound(); setIsInterviewOpen(false); }}
                className="flex items-center gap-1.5 px-3 py-1.5 border border-cyan-500/30 bg-cyan-950/50 hover:bg-cyan-900/80 text-cyan-300 rounded-lg transition-colors text-xs font-bold uppercase"
              >
                <X className="w-4 h-4" />
                <span className="hidden sm:inline">Exit Interview</span>
              </button>
            </div>

            {/* Main Content View */}
            {!showFinalReport ? (
              <div className="flex-1 w-full p-3 sm:p-6 overflow-y-auto grid grid-cols-1 lg:grid-cols-12 gap-5 max-w-7xl mx-auto">
                
                {/* LEFT COLUMN (6 COLS): AI HR INTERVIEWER & ANSWER AREA */}
                <div className="lg:col-span-6 flex flex-col gap-4">
                  
                  {/* AI HR Interviewer Avatar Box */}
                  <div className="bg-[#080d1a]/90 border border-cyan-500/40 rounded-xl p-4 sm:p-5 shadow-[0_0_30px_rgba(0,0,0,0.8)] relative overflow-hidden">
                    <div className="flex items-start justify-between gap-3 mb-3 pb-3 border-b border-cyan-500/20">
                      <div className="flex items-center gap-3">
                        <div className="relative">
                          <div className={`w-12 h-12 rounded-full bg-gradient-to-tr from-cyan-950 via-cyan-800 to-cyan-500 border-2 border-cyan-400 flex items-center justify-center text-white shadow-[0_0_20px_#00f0ff] ${isInterviewerSpeaking ? 'animate-bounce' : ''}`}>
                            <Users className="w-6 h-6 text-cyan-200" />
                          </div>
                          {isInterviewerSpeaking && (
                            <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-emerald-400 border-2 border-black rounded-full animate-ping" />
                          )}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <h3 className="font-bold text-sm text-cyan-200">SARAH VANCE</h3>
                            <span className="text-[9px] px-1.5 py-0.5 bg-cyan-950 border border-cyan-500/50 text-cyan-400 rounded">AI HR LEAD</span>
                          </div>
                          <p className="text-[10px] text-white/60">Senior Talent Acquisition & Executive Recruiter</p>
                        </div>
                      </div>

                      {/* Status pill */}
                      <div className="text-[10px] font-bold px-2.5 py-1 rounded-full border bg-black/60 flex items-center gap-1.5">
                        {isInterviewerSpeaking ? (
                          <span className="text-emerald-400 flex items-center gap-1">
                            <Volume2 className="w-3 h-3 animate-pulse" /> SPEAKING
                          </span>
                        ) : isEvaluatingAnswer ? (
                          <span className="text-cyan-400 flex items-center gap-1">
                            <Loader2 className="w-3 h-3 animate-spin" /> EVALUATING
                          </span>
                        ) : (
                          <span className="text-cyan-300 flex items-center gap-1">
                            <Mic className="w-3 h-3 text-cyan-400" /> LISTENING
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Question Content */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-[11px] text-cyan-400/80 uppercase font-bold">
                        <span>Topic: {GENERAL_HR_QUESTIONS[interviewStep].title}</span>
                        <button
                          onClick={() => speakHRQuestion(GENERAL_HR_QUESTIONS[interviewStep].question)}
                          className="flex items-center gap-1 px-2.5 py-1 bg-cyan-950 border border-cyan-400/50 hover:border-cyan-400 text-cyan-300 rounded text-[10px] font-bold transition-all shadow-[0_0_8px_rgba(0,240,255,0.2)]"
                        >
                          <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Listen Question</span>
                        </button>
                      </div>

                      <div className="p-3.5 bg-black/60 border border-cyan-500/30 rounded-lg">
                        <p className="text-xs sm:text-sm font-semibold text-white/95 leading-relaxed">
                          "{GENERAL_HR_QUESTIONS[interviewStep].question}"
                        </p>
                      </div>

                      <p className="text-[10px] text-cyan-400/70 italic flex items-center gap-1 pt-1">
                        <Sparkles className="w-3 h-3 text-cyan-400" />
                        <span>Advice: {GENERAL_HR_QUESTIONS[interviewStep].context}</span>
                      </p>
                    </div>
                  </div>

                  {/* Candidate Response Area */}
                  <div className="bg-[#080d1a]/90 border border-cyan-500/30 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between text-xs font-bold text-cyan-300 border-b border-cyan-500/20 pb-2">
                      <span className="flex items-center gap-1.5">
                        <MessageSquare className="w-4 h-4 text-cyan-400" /> YOUR RESPONSE
                      </span>
                      <span className="text-[10px] text-white/50">{userAnswerInput.length} chars</span>
                    </div>

                    <textarea
                      value={userAnswerInput}
                      onChange={(e) => setUserAnswerInput(e.target.value)}
                      placeholder="Type or click 'Mic Input' to speak your answer clearly... Use the STAR method (Situation, Task, Action, Result) for best evaluation."
                      rows={4}
                      className="w-full bg-black/60 border border-cyan-500/30 focus:border-cyan-400 rounded-lg p-3 text-xs text-white placeholder-white/30 focus:outline-none focus:ring-1 focus:ring-cyan-400 resize-none font-mono"
                    />

                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          onClick={toggleSpeechRecognition}
                          disabled={isTranscribing}
                          className={`flex items-center gap-2 px-3.5 py-2 rounded-lg border text-xs font-bold transition-all ${
                            isMicListening
                              ? 'bg-red-500/20 border-red-500 text-red-300 shadow-[0_0_15px_rgba(239,68,68,0.4)]'
                              : 'bg-cyan-950/80 border-cyan-500/40 text-cyan-300 hover:border-cyan-400'
                          }`}
                        >
                          <Mic className={`w-4 h-4 ${isMicListening ? 'animate-bounce text-red-400' : 'text-cyan-400'}`} />
                          <span>{isMicListening ? 'Listening (Click Stop)' : 'Mic Input'}</span>

                          {/* Live Volume Level Meter when active */}
                          {isMicListening && (
                            <div className="flex items-center gap-1.5 ml-1 px-2 py-0.5 bg-black/80 rounded border border-red-500/40 text-[10px]">
                              <div className="w-10 h-2 bg-black/90 rounded-full overflow-hidden border border-red-500/30">
                                <div 
                                  className="h-full bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500 transition-all duration-75"
                                  style={{ width: `${Math.max(8, micVolume)}%` }}
                                />
                              </div>
                              <span className="font-mono text-[9px] text-red-300">{micVolume}%</span>
                            </div>
                          )}
                        </button>

                        <button
                          onClick={() => {
                            const sample = GENERAL_HR_QUESTIONS[interviewStep]?.sampleAnswer || "";
                            setUserAnswerInput(sample);
                            playHoverSound();
                            setMicStatusText("Sample STAR answer inserted!");
                            setTimeout(() => setMicStatusText(""), 2500);
                          }}
                          className="flex items-center gap-1.5 px-3 py-2 bg-cyan-950/60 hover:bg-cyan-900/80 border border-cyan-500/30 text-cyan-300 rounded-lg text-xs font-semibold transition-all"
                        >
                          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
                          <span>Insert Sample STAR Answer</span>
                        </button>

                        {/* Mic Status Banner */}
                        {(micStatusText || isTranscribing) && (
                          <div className="w-full text-[10px] text-cyan-300 font-mono italic flex items-center gap-1 pt-1">
                            {isTranscribing && <Loader2 className="w-3 h-3 animate-spin text-cyan-400" />}
                            <span>{micStatusText}</span>
                          </div>
                        )}
                      </div>

                      <button
                        onClick={evaluateInterviewAnswerWithAI}
                        disabled={isEvaluatingAnswer || isTranscribing}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg font-bold text-xs uppercase tracking-wider shadow-[0_0_20px_rgba(0,240,255,0.3)] transition-all disabled:opacity-50"
                      >
                        {isEvaluatingAnswer ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Analyzing Biometrics & Text...</span>
                          </>
                        ) : (
                          <>
                            <Award className="w-4 h-4" />
                            <span>Submit Response for AI Feedback</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Previous Question Feedback Summary Cards */}
                  {interviewHistory.length > 0 && (
                    <div className="bg-[#080d1a]/80 border border-cyan-500/20 rounded-xl p-3.5 space-y-2.5 max-h-48 overflow-y-auto">
                      <span className="text-[10px] font-bold text-cyan-400/80 uppercase tracking-widest block">
                        Recent AI HR Feedback Logs ({interviewHistory.length})
                      </span>
                      {interviewHistory.slice().reverse().map((item, idx) => (
                        <div key={idx} className="p-2.5 bg-black/50 border border-cyan-500/20 rounded-lg space-y-1">
                          <div className="flex items-center justify-between text-[11px]">
                            <span className="font-bold text-white/90 truncate max-w-[240px]">{item.question}</span>
                            <div className="flex items-center gap-2">
                              <span className="text-emerald-400 font-bold">Verbal: {item.verbalScore}/100</span>
                              <span className="text-cyan-400 font-bold">Non-Verbal: {item.nonVerbalScore}/100</span>
                            </div>
                          </div>
                          <p className="text-[10px] text-white/70">{item.feedback}</p>
                          <p className="text-[9px] text-cyan-300 italic">💡 {item.tips}</p>
                        </div>
                      ))}
                    </div>
                  )}

                </div>

                {/* RIGHT COLUMN (6 COLS): CANDIDATE LIVE VIDEO & REAL-TIME EMOTION HUD */}
                <div className="lg:col-span-6 flex flex-col gap-4">
                  
                  {/* Camera Video View with Mesh Overlay (2x2 Expanded Aspect Ratio) */}
                  <div className="bg-[#080d1a] border-2 border-cyan-500/50 rounded-xl overflow-hidden relative shadow-[0_0_35px_rgba(0,240,255,0.25)] h-60 sm:h-80 lg:h-[480px] flex items-center justify-center transition-all">
                    <video
                      ref={interviewVideoRef}
                      autoPlay
                      playsInline
                      muted
                      className={`absolute inset-0 w-full h-full object-cover ${activeColorModeObj.cssFilter}`}
                    />
                    <canvas
                      ref={interviewCanvasRef}
                      className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10"
                    />

                    {/* Facial Emotion Overlay Badge */}
                    <div className="absolute top-3 left-3 z-20 px-3 py-1 bg-black/80 border border-cyan-400 rounded-lg text-xs font-bold text-cyan-300 shadow-[0_0_12px_#00f0ff] flex items-center gap-2">
                      <ScanFace className="w-4 h-4 text-cyan-400 animate-pulse" />
                      <span>EMOTION: {consoleState.emotion.toUpperCase()}</span>
                    </div>

                    <div className="absolute top-3 right-3 z-20 px-2.5 py-1 bg-cyan-950/90 border border-cyan-400 rounded text-[10px] font-bold text-cyan-300 shadow-[0_0_10px_rgba(0,240,255,0.3)]">
                      2X2 LARGE CAMERA :: LIVE ANALYTICS
                    </div>

                    <div className="absolute bottom-3 left-3 right-3 z-20 bg-black/85 border border-cyan-500/40 p-2.5 rounded-lg backdrop-blur-md flex items-center justify-between text-[11px] font-mono">
                      <span className="text-white/80">Smile Warmth: {Math.round((consoleState.blendshapes.smile || 0) * 100)}%</span>
                      <span className="text-cyan-400 font-bold">Frown Tension: {Math.round((consoleState.blendshapes.frown || 0) * 100)}%</span>
                    </div>
                  </div>

                  {/* Real-time Facial Telemetry Gauge Box */}
                  <div className="bg-[#080d1a]/90 border border-cyan-500/30 rounded-xl p-4 space-y-3">
                    <h4 className="text-xs font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5 border-b border-cyan-500/20 pb-2">
                      <Activity className="w-4 h-4 text-cyan-400" /> REAL-TIME CANDIDATE BIOMETRICS
                    </h4>

                    <div className="space-y-2.5 text-xs">
                      {/* Confidence Meter */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-white/70">Composure & Confidence</span>
                          <span className="text-cyan-300 font-bold">{calculateAffectiveMetrics().confidence}%</span>
                        </div>
                        <div className="w-full h-2 bg-black/60 rounded-full overflow-hidden border border-cyan-500/20">
                          <div 
                            className="h-full bg-gradient-to-r from-cyan-500 to-emerald-400 transition-all duration-300"
                            style={{ width: `${calculateAffectiveMetrics().confidence}%` }}
                          />
                        </div>
                      </div>

                      {/* Stress Index */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-white/70">Facial Stress & Tension</span>
                          <span className={`font-bold ${calculateAffectiveMetrics().stress > 40 ? 'text-rose-400' : 'text-emerald-400'}`}>
                            {calculateAffectiveMetrics().stress}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-black/60 rounded-full overflow-hidden border border-cyan-500/20">
                          <div 
                            className={`h-full transition-all duration-300 ${calculateAffectiveMetrics().stress > 40 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                            style={{ width: `${calculateAffectiveMetrics().stress}%` }}
                          />
                        </div>
                      </div>

                      {/* Eye Focus & Engagement */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-white/70">Eye Focus & Engagement</span>
                          <span className="text-cyan-300 font-bold">{calculateAffectiveMetrics().focus}%</span>
                        </div>
                        <div className="w-full h-2 bg-black/60 rounded-full overflow-hidden border border-cyan-500/20">
                          <div 
                            className="h-full bg-cyan-400 transition-all duration-300"
                            style={{ width: `${calculateAffectiveMetrics().focus}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Live Non-Verbal Coaching Tip Box */}
                    <div className="p-3 bg-cyan-950/40 border border-cyan-400/40 rounded-lg text-[11px] text-cyan-200 flex items-start gap-2">
                      <Sparkles className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-bold block text-cyan-300">Live Non-Verbal Coaching Tip:</span>
                        {(consoleState.blendshapes.smile || 0) > 0.35 ? (
                          <span>Excellent warmth! Maintain that relaxed smile while explaining your accomplishments.</span>
                        ) : (consoleState.blendshapes.frown || 0) > 0.25 ? (
                          <span>Brows show slight tension. Take a slow breath, soften your forehead, and speak steadily.</span>
                        ) : (
                          <span>Good neutral composure. Keep eye contact aligned with the camera for maximum connection.</span>
                        )}
                      </div>
                    </div>
                  </div>

                </div>
              </div>
            ) : (
              /* FINAL INTERVIEW SCORECARD REPORT SCREEN */
              <div className="flex-1 w-full p-4 sm:p-8 overflow-y-auto max-w-4xl mx-auto space-y-6">
                
                <div className="bg-[#080d1a] border-2 border-cyan-500/50 rounded-2xl p-6 sm:p-8 space-y-6 shadow-[0_0_50px_rgba(0,240,255,0.2)] text-center relative">
                  <div className="w-16 h-16 rounded-full bg-cyan-500/20 border-2 border-cyan-400 text-cyan-300 flex items-center justify-center mx-auto shadow-[0_0_20px_#00f0ff]">
                    <Award className="w-8 h-8" />
                  </div>

                  <div>
                    <h3 className="text-xl sm:text-2xl font-bold tracking-widest text-white">INTERVIEW EVALUATION COMPLETED</h3>
                    <p className="text-xs text-cyan-400 font-mono mt-1">GENERAL HR CANDIDATE SCORECARD & FACIAL BIOMETRICS SUMMARY</p>
                  </div>

                  {/* Overall Grade Banner */}
                  <div className="p-4 bg-black/60 border border-cyan-400/50 rounded-xl max-w-md mx-auto space-y-1">
                    <span className="text-[10px] text-cyan-400 uppercase font-bold tracking-wider">Hiring Recommendation</span>
                    <div className="text-2xl sm:text-3xl font-extrabold text-emerald-400 drop-shadow-[0_0_10px_#10b981]">
                      RECOMMENDED FOR HIRE (89 / 100)
                    </div>
                  </div>

                  {/* Biometric Progression Recharts Line & Area Chart */}
                  <div className="bg-black/60 border border-cyan-500/30 rounded-xl p-4 sm:p-5 text-left space-y-4 shadow-[0_0_20px_rgba(0,240,255,0.15)]">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-cyan-500/30 pb-3">
                      <div>
                        <h4 className="text-xs sm:text-sm font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-2">
                          <Activity className="w-4 h-4 text-cyan-400" /> STRESS, CONFIDENCE & VERBAL PROGRESSION
                        </h4>
                        <p className="text-[10px] text-white/60 font-mono mt-0.5">
                          Real-time facial telemetry tracking composure, stress & verbal score lines across interview steps
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono shrink-0">
                        <span className="flex items-center gap-1.5 text-cyan-300">
                          <span className="w-4 h-1 bg-cyan-400 rounded-full shadow-[0_0_8px_#00f0ff]" />
                          Confidence
                        </span>
                        <span className="flex items-center gap-1.5 text-rose-300">
                          <span className="w-4 h-1 bg-rose-500 rounded-full shadow-[0_0_8px_#f43f5e]" />
                          Stress Index
                        </span>
                        <span className="flex items-center gap-1.5 text-emerald-300">
                          <span className="w-4 h-1 bg-emerald-400 rounded-full shadow-[0_0_8px_#10b981]" />
                          Verbal Score
                        </span>
                      </div>
                    </div>

                    <div className="w-full h-64 sm:h-72">
                      {(() => {
                        const rawLogs = (interviewBiometricsLog.length > 0 
                          ? interviewBiometricsLog 
                          : interviewHistory.map((item, idx) => ({
                              step: idx + 1,
                              emotion: item.emotionAtAnswer,
                              confidence: item.nonVerbalScore,
                              stress: Math.max(10, 100 - item.nonVerbalScore),
                              positivity: Math.round(item.verbalScore)
                            }))
                        );

                        const formattedLogs = rawLogs.map((log, idx) => ({
                          stepLabel: `Question ${log.step || idx + 1}`,
                          shortLabel: `Q${log.step || idx + 1}`,
                          Confidence: log.confidence,
                          Stress: log.stress,
                          Verbal: interviewHistory[idx]?.verbalScore || 85,
                          Emotion: log.emotion || 'neutral',
                          QuestionText: interviewHistory[idx]?.question || `Question ${idx + 1}`
                        }));

                        const chartData = rawLogs.length > 0 ? [
                          {
                            stepLabel: 'Baseline Start',
                            shortLabel: 'Start',
                            Confidence: 70,
                            Stress: 30,
                            Verbal: 75,
                            Emotion: 'neutral',
                            QuestionText: 'Initial composure baseline before question evaluation'
                          },
                          ...formattedLogs
                        ] : [];

                        if (chartData.length === 0) {
                          return (
                            <div className="w-full h-full flex items-center justify-center text-xs text-cyan-400/60 font-mono border border-dashed border-cyan-500/20 rounded-lg">
                              No telemetry data recorded yet.
                            </div>
                          );
                        }

                        return (
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData} margin={{ top: 15, right: 20, left: -20, bottom: 5 }}>
                              <defs>
                                <linearGradient id="confidenceGrad" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#00f0ff" stopOpacity={0.35}/>
                                  <stop offset="95%" stopColor="#00f0ff" stopOpacity={0.0}/>
                                </linearGradient>
                                <linearGradient id="stressGrad" x1="0" y1="0" x2="0" y2="1">
                                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.35}/>
                                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0}/>
                                </linearGradient>
                              </defs>
                              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,240,255,0.15)" />
                              <XAxis 
                                dataKey="shortLabel" 
                                stroke="#00f0ff" 
                                tick={{ fill: '#a5f3fc', fontSize: 11, fontFamily: 'monospace' }} 
                              />
                              <YAxis 
                                domain={[0, 100]} 
                                stroke="#00f0ff" 
                                tick={{ fill: '#a5f3fc', fontSize: 11, fontFamily: 'monospace' }} 
                                unit="%" 
                              />
                              <Tooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                      <div className="bg-[#080d1a]/95 border border-cyan-400 p-3 rounded-lg shadow-[0_0_20px_rgba(0,240,255,0.3)] backdrop-blur-md text-xs font-mono space-y-1.5 max-w-xs">
                                        <p className="font-bold text-cyan-300 border-b border-cyan-500/30 pb-1">
                                          {data.shortLabel}: {data.QuestionText}
                                        </p>
                                        <div className="flex items-center justify-between gap-3 text-cyan-200">
                                          <span>Confidence Level:</span>
                                          <span className="font-bold text-cyan-300">{data.Confidence}%</span>
                                        </div>
                                        <div className="flex items-center justify-between gap-3 text-rose-200">
                                          <span>Stress Index:</span>
                                          <span className="font-bold text-rose-400">{data.Stress}%</span>
                                        </div>
                                        <div className="flex items-center justify-between gap-3 text-emerald-200">
                                          <span>Verbal Score:</span>
                                          <span className="font-bold text-emerald-400">{data.Verbal}%</span>
                                        </div>
                                        <div className="flex items-center justify-between gap-3 text-amber-200 text-[10px]">
                                          <span>Emotion Detected:</span>
                                          <span className="font-bold uppercase text-amber-300">{data.Emotion}</span>
                                        </div>
                                      </div>
                                    );
                                  }
                                  return null;
                                }}
                              />
                              <Legend 
                                wrapperStyle={{ fontSize: '11px', fontFamily: 'monospace', paddingTop: '10px' }} 
                              />
                              <Area
                                type="monotone"
                                dataKey="Confidence"
                                name="Confidence Level (%)"
                                stroke="#00f0ff"
                                strokeWidth={3.5}
                                fillOpacity={1}
                                fill="url(#confidenceGrad)"
                                connectNulls={true}
                                isAnimationActive={false}
                                dot={{ r: 5, fill: '#00f0ff', stroke: '#080d1a', strokeWidth: 2 }}
                                activeDot={{ r: 8, fill: '#00f0ff', stroke: '#ffffff', strokeWidth: 2 }}
                              />
                              <Area
                                type="monotone"
                                dataKey="Stress"
                                name="Stress Level (%)"
                                stroke="#f43f5e"
                                strokeWidth={3.5}
                                fillOpacity={1}
                                fill="url(#stressGrad)"
                                connectNulls={true}
                                isAnimationActive={false}
                                dot={{ r: 5, fill: '#f43f5e', stroke: '#080d1a', strokeWidth: 2 }}
                                activeDot={{ r: 8, fill: '#f43f5e', stroke: '#ffffff', strokeWidth: 2 }}
                              />
                              <Line
                                type="monotone"
                                dataKey="Verbal"
                                name="Verbal Clarity Score (%)"
                                stroke="#10b981"
                                strokeWidth={3}
                                connectNulls={true}
                                isAnimationActive={false}
                                dot={{ r: 4, fill: '#10b981', stroke: '#080d1a', strokeWidth: 2 }}
                              />
                            </AreaChart>
                          </ResponsiveContainer>
                        );
                      })()}
                    </div>
                  </div>

                  {/* Question Breakdown Table */}
                  <div className="text-left space-y-3 pt-2">
                    <h4 className="text-xs font-bold text-cyan-300 uppercase tracking-wider flex items-center gap-2 border-b border-cyan-500/30 pb-2">
                      <BarChart2 className="w-4 h-4 text-cyan-400" /> QUESTION PERFORMANCE BREAKDOWN
                    </h4>

                    <div className="space-y-3">
                      {interviewHistory.map((item, idx) => (
                        <div key={idx} className="p-4 bg-black/50 border border-cyan-500/20 rounded-xl space-y-2 text-xs">
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-1 border-b border-white/10">
                            <span className="font-bold text-cyan-200">Q{idx + 1}: {item.question}</span>
                            <div className="flex items-center gap-3 text-xs">
                              <span className="text-emerald-400 font-bold">Verbal: {item.verbalScore}/100</span>
                              <span className="text-cyan-400 font-bold">Emotion: {item.nonVerbalScore}/100 ({item.emotionAtAnswer})</span>
                            </div>
                          </div>
                          <p className="text-white/80 text-xs">{item.feedback}</p>
                          <p className="text-cyan-300 text-[11px] italic">💡 Key Tip: {item.tips}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
                    <button
                      onClick={() => {
                        setInterviewStep(0);
                        setInterviewHistory([]);
                        setInterviewBiometricsLog([]);
                        setShowFinalReport(false);
                        setUserAnswerInput('');
                      }}
                      className="w-full sm:w-auto px-6 py-2.5 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-400 text-cyan-300 rounded-xl font-bold text-xs uppercase tracking-wider transition-all"
                    >
                      Restart Interview
                    </button>

                    <button
                      onClick={() => {
                        setIsInterviewOpen(false);
                        setShowFinalReport(false);
                      }}
                      className="w-full sm:w-auto px-8 py-2.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-xl font-bold text-xs uppercase tracking-wider shadow-[0_0_20px_rgba(0,240,255,0.3)] transition-all"
                    >
                      Return to Vision HUD
                    </button>
                  </div>

                </div>

              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
