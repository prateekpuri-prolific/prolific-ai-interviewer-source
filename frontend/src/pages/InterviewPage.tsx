/**
 * OASIS — Participant interview widget.
 *
 * Accessible at /interview/:widgetKey — this is the page a study participant
 * sees when they join a voice or text interview via the shareable link.
 *
 * Features:
 *  - Fetches widget config (title, description, colour, participant ID mode, modality)
 *  - Handles participant identification: random / predefined (via ?pid=) / user input
 *  - Voice mode: text-first question display with spoken audio and live status
 *  - Text mode: modern chat interface with avatars and typing indicators
 *  - Call timer
 *  - Backend logs diarised transcripts for researcher review
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { widget, type WidgetConfig } from "../lib/api";
import { hexToRgba, hexToRgb } from "../lib/colors";
import { MicCapture, AudioPlayer, SpeechEndpointDetector } from "../lib/audio";
import { encodeAudioFrame, decodeFrame } from "../lib/pipecat-proto";

type Status = "loading" | "idle" | "input" | "connecting" | "active" | "ended" | "error";
type EndedReason = "natural" | "user_stopped" | "inactive";
type ResponseMode = "voice" | "text";
type RecordingState = "idle" | "recording" | "confirming" | "transcribing";
type MicCheckStatus = "idle" | "checking" | "ready" | "quiet" | "denied" | "unavailable";

interface ChatMessage {
  id: string;
  role: "agent" | "user";
  text: string;
  timestamp: Date;
}

/**
 * Lightweight markdown-to-HTML renderer for chat messages.
 * Handles: **bold**, *italic*, `inline code`, [links](url), line breaks,
 * unordered lists (- item), ordered lists (1. item).
 * Sanitises HTML entities to prevent XSS.
 */
function renderMarkdown(text: string): string {
  // Escape HTML entities first
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks (``` ... ```)
  html = html.replace(/```([\s\S]*?)```/g, '<pre class="bg-black/5 rounded-lg p-2 my-1 text-xs font-mono overflow-x-auto whitespace-pre-wrap">$1</pre>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code class="bg-black/10 rounded px-1 py-0.5 text-xs font-mono">$1</code>');

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic (single asterisk, not inside a word)
  html = html.replace(/(?<!\w)\*(.+?)\*(?!\w)/g, "<em>$1</em>");

  // Links [text](url)
  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer" class="underline hover:opacity-80">$1</a>'
  );

  // Process lines for lists
  const lines = html.split("\n");
  const result: string[] = [];
  let inUl = false;
  let inOl = false;

  for (const line of lines) {
    const ulMatch = line.match(/^\s*[-•]\s+(.+)/);
    const olMatch = line.match(/^\s*\d+[.)]\s+(.+)/);

    if (ulMatch) {
      if (!inUl) { result.push('<ul class="list-disc list-inside my-1 space-y-0.5">'); inUl = true; }
      if (inOl) { result.push("</ol>"); inOl = false; }
      result.push(`<li>${ulMatch[1]}</li>`);
    } else if (olMatch) {
      if (!inOl) { result.push('<ol class="list-decimal list-inside my-1 space-y-0.5">'); inOl = true; }
      if (inUl) { result.push("</ul>"); inUl = false; }
      result.push(`<li>${olMatch[1]}</li>`);
    } else {
      if (inUl) { result.push("</ul>"); inUl = false; }
      if (inOl) { result.push("</ol>"); inOl = false; }
      result.push(line);
    }
  }
  if (inUl) result.push("</ul>");
  if (inOl) result.push("</ol>");

  // Join lines, convert remaining newlines to <br>
  html = result.join("\n").replace(/\n/g, "<br>");
  // Clean up double <br> from list boundaries
  html = html.replace(/<br><(ul|ol)/g, "<$1");
  html = html.replace(/<\/(ul|ol)><br>/g, "</$1>");

  return html;
}

/** Chat bubble content component that renders markdown safely */
function ChatBubbleContent({ text, isUser }: { text: string; isUser: boolean }) {
  const html = renderMarkdown(text);
  return (
    <div
      className={isUser ? "chat-content-user" : "chat-content-agent"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function safeHandoffUrl(value?: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.toString() : null;
  } catch {
    return null;
  }
}

export default function InterviewPage() {
  const { widgetKey } = useParams<{ widgetKey: string }>();
  const [searchParams] = useSearchParams();
  const pidFromUrl = searchParams.get("pid") || searchParams.get("PROLIFIC_PID");
  const externalStudyId = searchParams.get("STUDY_ID") || searchParams.get("study_id");
  const externalSessionId = searchParams.get("SESSION_ID") || searchParams.get("session_id");

  const [status, setStatus] = useState<Status>("loading");
  const [endedReason, setEndedReason] = useState<EndedReason>("natural");
  const [errorMsg, setErrorMsg] = useState("");
  const [config, setConfig] = useState<WidgetConfig | null>(null);
  const [participantId, setParticipantId] = useState(pidFromUrl || "");
  const [consentAccepted, setConsentAccepted] = useState(false);
  const [agentSpeaking, setAgentSpeaking] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [voicePrompt, setVoicePrompt] = useState("");
  const [voiceUserText, setVoiceUserText] = useState("");
  const [audioEnabled, setAudioEnabled] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState("");
  const [responseMode, setResponseMode] = useState<ResponseMode>("voice");
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingError, setRecordingError] = useState("");
  const [finishDialogOpen, setFinishDialogOpen] = useState(false);
  const [finishingInterview, setFinishingInterview] = useState(false);
  const [inactivityWarning, setInactivityWarning] = useState(false);
  const [inactivitySecondsRemaining, setInactivitySecondsRemaining] = useState(60);
  const [returnUrl, setReturnUrl] = useState<string | null>(null);
  const [micCheckStatus, setMicCheckStatus] = useState<MicCheckStatus>("idle");
  const [micCheckMessage, setMicCheckMessage] = useState("");

  // ── Text chat state ──
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [agentTyping, setAgentTyping] = useState(false);
  const latestPromptRef = useRef<HTMLElement>(null);
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  // Grow the chat input with its content up to a cap, then scroll inside it.
  const MAX_INPUT_HEIGHT = 160; // px, matches max-h-40
  const resizeChatInput = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_INPUT_HEIGHT)}px`;
  }, []);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const promptAudioRef = useRef<HTMLAudioElement | null>(null);
  const promptAudioUrlRef = useRef<string | null>(null);
  const promptAudioAbortRef = useRef<AbortController | null>(null);
  const audioEnabledRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const speakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const userSpeakingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inactivityCountdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastActivitySignalRef = useRef(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const endpointAudioContextRef = useRef<AudioContext | null>(null);
  const endpointFrameRef = useRef<number | null>(null);
  const recordingStateRef = useRef<RecordingState>("idle");
  const responseModeRef = useRef<ResponseMode>("voice");
  const agentTypingRef = useRef(false);
  const startRecordingRef = useRef<(() => Promise<void>) | null>(null);
  const finishAfterCurrentAnswerRef = useRef(false);
  const finishingInterviewRef = useRef(false);
  const returnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const micCheckStreamRef = useRef<MediaStream | null>(null);
  const micCheckContextRef = useRef<AudioContext | null>(null);
  const micCheckFrameRef = useRef<number | null>(null);

  useEffect(() => {
    recordingStateRef.current = recordingState;
  }, [recordingState]);

  useEffect(() => {
    responseModeRef.current = responseMode;
  }, [responseMode]);

  useEffect(() => {
    agentTypingRef.current = agentTyping;
  }, [agentTyping]);

  // Derived
  // Listen-style participant delivery is text-first for every modular agent.
  // The agent's legacy modality still controls its available provider config,
  // but never turns the participant microphone on automatically.
  const isTextChat = true;
  const primaryColor = config?.widget_primary_color || "#0D7377";
  const listeningMessage = config?.widget_listening_message || "Agent is listening…";
  const showProgress =
    !!config?.widget_show_progress &&
    config?.interview_mode === "structured" &&
    (config?.question_count ?? 0) > 0;
  // Guide position is only a proxy for elapsed interview time because each
  // card can have a different number of follow-ups. Round to broad 10-point
  // steps and stop at 90% until the interview actually ends so the UI never
  // implies a precision or finish time the moderator cannot guarantee.
  const approximateProgressPercent = progress
    ? Math.min(
        90,
        Math.max(
          10,
          Math.round(
            (progress.current / Math.max(1, progress.total)) * 10,
          ) * 10,
        ),
      )
    : 5;
  const approximateProgressLabel = progress
    ? `About ${approximateProgressPercent}% complete`
    : "Getting started";
  const immersiveInterview = status === "active" || status === "connecting";
  const participantInfo = config?.participant_information;
  const canStart = !participantInfo?.consent_required || consentAccepted;

  // ── 1. Fetch widget config on mount ─────────────────────────
  useEffect(() => {
    if (!widgetKey) return;
    widget
      .config(widgetKey)
      .then((cfg) => {
        setConfig(cfg);
        const defaultAudioEnabled = !!cfg.audio_default_on;
        audioEnabledRef.current = defaultAudioEnabled;
        setAudioEnabled(defaultAudioEnabled);
        const defaultResponseMode =
          cfg.voice_response_available && cfg.response_mode_default === "voice"
            ? "voice"
            : "text";
        responseModeRef.current = defaultResponseMode;
        setResponseMode(defaultResponseMode);
        if (cfg.participant_id_mode === "predefined" || cfg.participant_id_mode === "external") {
          if (!pidFromUrl) {
            setStatus("error");
            setErrorMsg(
              "This interview requires a valid participant link. Please use the link provided by your researcher."
            );
          } else if (
            cfg.participant_id_mode === "external"
            && cfg.fielding?.recruitment_source === "prolific"
            && !externalSessionId
          ) {
            setStatus("error");
            setErrorMsg("This Prolific link is incomplete. Please reopen the study from Prolific.");
          } else {
            setParticipantId(pidFromUrl);
            setStatus("idle");
          }
        } else if (cfg.participant_id_mode === "input") {
          setStatus(pidFromUrl ? "idle" : "input");
        } else {
          setStatus("idle");
        }
      })
      .catch(() => {
        setStatus("error");
        setErrorMsg("This interview is not available. Please check the link or contact your researcher.");
      });
  }, [widgetKey, pidFromUrl, externalSessionId]);

  // Keep the start of the current interviewer turn visible. Scrolling to the
  // transcript bottom clips longer prompts above the viewport.
  useEffect(() => {
    latestPromptRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [chatMessages, agentTyping]);

  const stopPromptAudio = useCallback(() => {
    promptAudioAbortRef.current?.abort();
    promptAudioAbortRef.current = null;
    if (promptAudioRef.current) {
      promptAudioRef.current.pause();
      promptAudioRef.current.currentTime = 0;
      promptAudioRef.current = null;
    }
    if (promptAudioUrlRef.current) {
      URL.revokeObjectURL(promptAudioUrlRef.current);
      promptAudioUrlRef.current = null;
    }
    setAudioLoading(false);
  }, []);

  const playPromptAudio = useCallback(async (text: string, force = false) => {
    if (!widgetKey || !config?.audio_available || (!audioEnabledRef.current && !force) || !text.trim()) return;
    stopPromptAudio();
    setAudioError("");
    setAudioLoading(true);
    const controller = new AbortController();
    promptAudioAbortRef.current = controller;

    try {
      const response = await fetch(`/api/widget/${widgetKey}/speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("Audio is temporarily unavailable");
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      promptAudioUrlRef.current = url;
      const audio = new Audio(url);
      promptAudioRef.current = audio;
      audio.onended = () => {
        if (promptAudioUrlRef.current === url) {
          URL.revokeObjectURL(url);
          promptAudioUrlRef.current = null;
          promptAudioRef.current = null;
        }
        if (responseModeRef.current === "voice") {
          void startRecordingRef.current?.();
        }
      };
      await audio.play();
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setAudioError("Audio is temporarily unavailable. You can continue in text.");
        if (responseModeRef.current === "voice") {
          void startRecordingRef.current?.();
        }
      }
    } finally {
      if (promptAudioAbortRef.current === controller) promptAudioAbortRef.current = null;
      setAudioLoading(false);
    }
  }, [config?.audio_available, stopPromptAudio, widgetKey]);

  const cleanupTurnRecording = useCallback(() => {
    if (endpointFrameRef.current != null) {
      cancelAnimationFrame(endpointFrameRef.current);
      endpointFrameRef.current = null;
    }
    if (endpointAudioContextRef.current) {
      void endpointAudioContextRef.current.close().catch(() => undefined);
      endpointAudioContextRef.current = null;
    }
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    mediaRecorderRef.current = null;
  }, []);

  const releaseTurnMicrophone = useCallback(() => {
    cleanupTurnRecording();
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
  }, [cleanupTurnRecording]);

  const cleanupConnection = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    micRef.current?.stop();
    micRef.current = null;
    playerRef.current?.stop();
    stopPromptAudio();
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.onstop = null;
      mediaRecorderRef.current.stop();
    }
    releaseTurnMicrophone();
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (speakingTimeoutRef.current) {
      clearTimeout(speakingTimeoutRef.current);
      speakingTimeoutRef.current = null;
    }
    if (userSpeakingTimeoutRef.current) {
      clearTimeout(userSpeakingTimeoutRef.current);
      userSpeakingTimeoutRef.current = null;
    }
    if (inactivityCountdownRef.current) {
      clearInterval(inactivityCountdownRef.current);
      inactivityCountdownRef.current = null;
    }
  }, [releaseTurnMicrophone, stopPromptAudio]);

  const clearInactivityWarning = useCallback(() => {
    setInactivityWarning(false);
    if (inactivityCountdownRef.current) {
      clearInterval(inactivityCountdownRef.current);
      inactivityCountdownRef.current = null;
    }
  }, []);

  const signalParticipantActivity = useCallback((force = false) => {
    const now = Date.now();
    if (
      !force &&
      now - lastActivitySignalRef.current < 15_000
    ) return;
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "keepalive" }));
      lastActivitySignalRef.current = now;
    }
  }, []);

  const pauseRecordingForInactivity = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === "recording") {
      recorder.onstop = null;
      recorder.stop();
    }
    recordingChunksRef.current = [];
    releaseTurnMicrophone();
    recordingStateRef.current = "idle";
    setRecordingState("idle");
    setRecordingSeconds(0);
    setRecordingError("");
  }, [releaseTurnMicrophone]);

  const showInactivityWarning = useCallback((seconds: number) => {
    pauseRecordingForInactivity();
    const initialSeconds = Math.max(1, seconds || 60);
    setInactivityWarning(true);
    setInactivitySecondsRemaining(initialSeconds);
    if (inactivityCountdownRef.current) {
      clearInterval(inactivityCountdownRef.current);
    }
    inactivityCountdownRef.current = setInterval(() => {
      setInactivitySecondsRemaining((current) => Math.max(0, current - 1));
    }, 1000);
  }, [pauseRecordingForInactivity]);

  const resumeAfterInactivity = useCallback(() => {
    signalParticipantActivity(true);
    clearInactivityWarning();
    if (responseModeRef.current === "voice") {
      window.setTimeout(() => void startRecordingRef.current?.(), 0);
    }
  }, [clearInactivityWarning, signalParticipantActivity]);

  const cleanupMicCheck = useCallback(() => {
    if (micCheckFrameRef.current !== null) {
      cancelAnimationFrame(micCheckFrameRef.current);
      micCheckFrameRef.current = null;
    }
    micCheckStreamRef.current?.getTracks().forEach((track) => track.stop());
    micCheckStreamRef.current = null;
    const context = micCheckContextRef.current;
    micCheckContextRef.current = null;
    if (context && context.state !== "closed") void context.close();
  }, []);

  const runMicrophoneCheck = useCallback(async () => {
    cleanupMicCheck();
    setMicCheckStatus("checking");
    setMicCheckMessage("Checking your microphone… say a few words.");
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicCheckStatus("unavailable");
      setMicCheckMessage("This browser cannot run a microphone check. You can answer by typing instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micCheckStreamRef.current = stream;
      const context = new AudioContext();
      micCheckContextRef.current = context;
      if (context.state === "suspended") await context.resume();
      const analyser = context.createAnalyser();
      analyser.fftSize = 1024;
      context.createMediaStreamSource(stream).connect(analyser);
      const samples = new Float32Array(analyser.fftSize);
      const startedAt = performance.now();
      let peak = 0;
      const sample = () => {
        analyser.getFloatTimeDomainData(samples);
        let sum = 0;
        for (const value of samples) sum += value * value;
        peak = Math.max(peak, Math.sqrt(sum / samples.length));
        if (performance.now() - startedAt < 2500) {
          micCheckFrameRef.current = requestAnimationFrame(sample);
          return;
        }
        cleanupMicCheck();
        if (peak > 0.012) {
          setMicCheckStatus("ready");
          setMicCheckMessage("Microphone connected and sound detected.");
        } else {
          setMicCheckStatus("quiet");
          setMicCheckMessage("Your microphone connected, but we did not hear sound. Check that the correct input is selected and not muted, then try again—or answer by typing.");
        }
      };
      micCheckFrameRef.current = requestAnimationFrame(sample);
    } catch (error) {
      cleanupMicCheck();
      const denied = error instanceof DOMException && (error.name === "NotAllowedError" || error.name === "SecurityError");
      setMicCheckStatus(denied ? "denied" : "unavailable");
      setMicCheckMessage(
        denied
          ? "Microphone access is blocked. Open this site's permissions from the address bar, set Microphone to Allow, then retry—or answer by typing."
          : "We could not access a microphone. Check your device input, then retry—or answer by typing.",
      );
    }
  }, [cleanupMicCheck]);

  // ── Cleanup on unmount ──────────────────────────────────────
  useEffect(() => {
    return () => {
      cleanupConnection();
      cleanupMicCheck();
      playerRef.current?.destroy();
      if (returnTimerRef.current) clearTimeout(returnTimerRef.current);
    };
  }, [cleanupConnection, cleanupMicCheck]);

  // ── Audio level handler (for voice wave) ────────────────────
  const handleAudioLevel = useCallback((level: number) => {
    setAudioLevel(level);

    if (level > 0.05) {
      setUserSpeaking(true);
      if (userSpeakingTimeoutRef.current) clearTimeout(userSpeakingTimeoutRef.current);
      userSpeakingTimeoutRef.current = setTimeout(() => setUserSpeaking(false), 400);
    }
  }, []);

  // ── 2. Start interview ──────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (!widgetKey || !canStart) return;
    cleanupConnection();
    setStatus("connecting");
    setErrorMsg("");
    setElapsed(0);
    setReturnUrl(null);
    setEndedReason("natural");
    setProgress(null);
    setInactivityWarning(false);
    setInactivitySecondsRemaining(60);
    setVoicePrompt(config?.welcome_message || "Thanks for joining. Let’s begin.");
    setVoiceUserText("");

    try {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      let wsUrl = `${proto}//${window.location.host}/ws/interview/${widgetKey}`;

      const pid =
        config?.participant_id_mode === "random"
          ? undefined
          : participantId.trim() || undefined;
      const params = new URLSearchParams();
      if (pid) params.set("pid", pid);
      if (externalStudyId) params.set("external_study_id", externalStudyId);
      if (externalSessionId) params.set("external_session_id", externalSessionId);
      if (consentAccepted) params.set("consent", "true");
      params.set("response_mode", responseModeRef.current);
      if (micCheckStatus !== "checking") params.set("mic_check_status", micCheckStatus);
      if (params.size) wsUrl += `?${params.toString()}`;

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      const player = new AudioPlayer();
      playerRef.current = player;

      // Track whether we received a server-side error so ws.onclose
      // doesn't overwrite the error state with "ended" (race condition).
      let receivedError = false;

      ws.onopen = async () => {
        const start = Date.now();
        timerRef.current = setInterval(() => {
          setElapsed(Math.floor((Date.now() - start) / 1000));
        }, 1000);

        const mic = new MicCapture();
        micRef.current = mic;

        try {
          await mic.start(
            (pcm16) => {
              if (ws.readyState === WebSocket.OPEN) {
                const frame = encodeAudioFrame(pcm16);
                ws.send(frame);
              }
            },
            handleAudioLevel
          );
          setStatus("active");
        } catch (micErr) {
          receivedError = true;
          ws.close();
          if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
          }
          micRef.current?.stop();
          micRef.current = null;
          const name = micErr instanceof DOMException ? micErr.name : "";
          setStatus("error");
          setErrorMsg(
            name === "NotAllowedError" || name === "PermissionDeniedError"
              ? "Microphone access is required for this interview. Please allow the microphone in your browser settings and try again."
              : "Could not access your microphone. Please check your device and try again."
          );
        }
      };

      ws.onmessage = (event) => {
        const data =
          event.data instanceof ArrayBuffer
            ? new Uint8Array(event.data)
            : null;

        if (!data) {
          try {
            const json = JSON.parse(event.data as string);
            if (json.error) {
              receivedError = true;
              setStatus("error");
              setErrorMsg(json.error);
            } else if (json.type === "progress") {
              setProgress({ current: json.current, total: json.total });
            } else if (json.type === "transcript") {
              if (json.role === "agent") setVoicePrompt(json.content || "");
              if (json.role === "user") setVoiceUserText(json.content || "");
            }
          } catch {
            /* ignore */
          }
          return;
        }

        const decoded = decodeFrame(data);

        switch (decoded.type) {
          case "audio":
            setAgentSpeaking(true);
            player.play(decoded.audio, decoded.sampleRate);
            if (speakingTimeoutRef.current)
              clearTimeout(speakingTimeoutRef.current);
            speakingTimeoutRef.current = setTimeout(
              () => setAgentSpeaking(false),
              600
            );
            break;
        }
      };

      ws.onclose = (event) => {
        micRef.current?.stop();
        if (timerRef.current) clearInterval(timerRef.current);

        // Don't overwrite error status — the error message is more useful
        // than a generic "Interview complete" screen.
        if (receivedError) return;

        // WebSocket close codes 4000+ are application-level errors
        if (event.code >= 4000) {
          setStatus("error");
          setErrorMsg(
            event.reason || "The interview could not be started. Please check the link or contact your researcher."
          );
          return;
        }

        setEndedReason("natural");
        setStatus("ended");
      };

      ws.onerror = () => {
        receivedError = true;
        setStatus("error");
        setErrorMsg("Connection lost. Please try again.");
      };
    } catch (err) {
      setStatus("error");
      setErrorMsg(
        err instanceof Error ? err.message : "Failed to start interview"
      );
    }
  }, [widgetKey, config, participantId, externalStudyId, externalSessionId, consentAccepted, canStart, handleAudioLevel, cleanupConnection, micCheckStatus]);

  // ── 2b. Start text chat ─────────────────────────────────────
  const handleStartChat = useCallback(async () => {
    if (!widgetKey || !canStart) return;
    cleanupConnection();
    setStatus("connecting");
    setErrorMsg("");
    setElapsed(0);
    setReturnUrl(null);
    setEndedReason("natural");
    setProgress(null);
    setInactivityWarning(false);
    setInactivitySecondsRemaining(60);
    setChatMessages([]);
    agentTypingRef.current = false;
    setAgentTyping(false);
    recordingStateRef.current = "idle";
    setRecordingState("idle");
    setRecordingSeconds(0);
    setRecordingError("");
    setFinishDialogOpen(false);
    setFinishingInterview(false);
    finishingInterviewRef.current = false;
    finishAfterCurrentAnswerRef.current = false;

    try {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      let wsUrl = `${proto}//${window.location.host}/ws/chat/${widgetKey}`;

      const pid =
        config?.participant_id_mode === "random"
          ? undefined
          : participantId.trim() || undefined;
      const params = new URLSearchParams();
      if (pid) params.set("pid", pid);
      if (externalStudyId) params.set("external_study_id", externalStudyId);
      if (externalSessionId) params.set("external_session_id", externalSessionId);
      if (consentAccepted) params.set("consent", "true");
      params.set("response_mode", responseModeRef.current);
      if (micCheckStatus !== "checking") params.set("mic_check_status", micCheckStatus);
      if (params.size) wsUrl += `?${params.toString()}`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("active");
        const start = Date.now();
        timerRef.current = setInterval(() => {
          setElapsed(Math.floor((Date.now() - start) / 1000));
        }, 1000);
      };

      let receivedError = false;

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data as string);
          switch (msg.type) {
            case "welcome":
              // The server includes the agent's avatar in the payload; use it
              // to keep the chat in sync if the agent was edited mid-flight.
              if (msg.avatar && msg.avatar !== config?.avatar) {
                setConfig((prev) => (prev ? { ...prev, avatar: msg.avatar } : prev));
              }
              setChatMessages((prev) => [
                ...prev,
                { id: crypto.randomUUID(), role: "agent", text: msg.text, timestamp: new Date() },
              ]);
              agentTypingRef.current = false;
              setAgentTyping(false);
              if (audioEnabledRef.current && config?.audio_available) {
                void playPromptAudio(msg.text);
              } else if (responseModeRef.current === "voice") {
                window.setTimeout(() => void startRecordingRef.current?.(), 80);
              }
              break;
            case "message":
              {
                const messageRole = msg.role || "agent";
              setChatMessages((prev) => [
                ...prev,
                { id: crypto.randomUUID(), role: messageRole, text: msg.text, timestamp: new Date() },
              ]);
              agentTypingRef.current = false;
              setAgentTyping(false);
              if (messageRole === "agent") {
                if (audioEnabledRef.current && config?.audio_available) {
                  void playPromptAudio(msg.text);
                } else if (responseModeRef.current === "voice") {
                  window.setTimeout(() => void startRecordingRef.current?.(), 80);
                }
              }
              }
              break;
            case "typing":
              agentTypingRef.current = true;
              setAgentTyping(true);
              break;
            case "progress":
              setProgress({ current: msg.current, total: msg.total });
              break;
            case "inactivity_warning":
              showInactivityWarning(Number(msg.seconds_remaining) || 60);
              break;
            case "ping":
              break;
            case "ended":
              releaseTurnMicrophone();
              clearInactivityWarning();
              setEndedReason(
                msg.reason === "participant_finished"
                  ? "user_stopped"
                  : msg.reason === "participant_inactive"
                    ? "inactive"
                    : "natural"
              );
              setStatus("ended");
              setFinishingInterview(false);
              finishingInterviewRef.current = false;
              agentTypingRef.current = false;
              setAgentTyping(false);
              break;
            case "finalized": {
              const inactive = ["participant_inactive", "disconnected", "cancelled"].includes(msg.reason);
              const target = safeHandoffUrl(
                inactive
                  ? config?.fielding?.inactivity_url
                  : config?.fielding?.completion_url,
              );
              if (target) {
                setReturnUrl(target);
                if (returnTimerRef.current) clearTimeout(returnTimerRef.current);
                returnTimerRef.current = setTimeout(() => window.location.assign(target), 1400);
              }
              break;
            }
            case "error":
              receivedError = true;
              releaseTurnMicrophone();
              setStatus("error");
              setErrorMsg(msg.text);
              agentTypingRef.current = false;
              setAgentTyping(false);
              break;
          }
        } catch {
          /* ignore */
        }
      };

      ws.onclose = (event) => {
        releaseTurnMicrophone();
        if (timerRef.current) clearInterval(timerRef.current);
        if (receivedError) return;
        if (event.code >= 4000) {
          setStatus("error");
          setErrorMsg(event.reason || "The interview could not be started.");
          return;
        }
        setEndedReason(finishingInterviewRef.current ? "user_stopped" : "natural");
        setStatus("ended");
        setFinishingInterview(false);
        finishingInterviewRef.current = false;
      };

      ws.onerror = () => {
        receivedError = true;
        releaseTurnMicrophone();
        setStatus("error");
        setErrorMsg("Connection lost. Please try again.");
      };
    } catch (err) {
      setStatus("error");
      setErrorMsg(err instanceof Error ? err.message : "Failed to start");
    }
  }, [widgetKey, config, participantId, externalStudyId, externalSessionId, consentAccepted, canStart, cleanupConnection, playPromptAudio, releaseTurnMicrophone, showInactivityWarning, clearInactivityWarning, micCheckStatus]);

  const sendParticipantText = useCallback((value: string, finishInterview = false) => {
    const text = value.trim();
    if ((!text && !finishInterview) || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false;

    if (text) {
      setChatMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", text, timestamp: new Date() },
      ]);
    }
    setChatInput("");
    recordingStateRef.current = "idle";
    setRecordingState("idle");
    setRecordingSeconds(0);
    setRecordingError("");
    clearInactivityWarning();
    agentTypingRef.current = !finishInterview;
    setAgentTyping(!finishInterview);
    if (chatInputRef.current) chatInputRef.current.style.height = "auto";
    if (finishInterview) {
      finishingInterviewRef.current = true;
      setFinishingInterview(true);
      wsRef.current.send(JSON.stringify({ type: "finish", text }));
    } else {
      wsRef.current.send(JSON.stringify({ type: "message", text }));
    }
    return true;
  }, [clearInactivityWarning]);

  const cancelRecording = useCallback(() => {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === "recording") {
      recorder.onstop = null;
      recorder.stop();
    }
    recordingChunksRef.current = [];
    releaseTurnMicrophone();
    recordingStateRef.current = "idle";
    setRecordingState("idle");
    setRecordingSeconds(0);
    setRecordingError("");
    setChatInput("");
  }, [releaseTurnMicrophone]);

  const startEndpointMonitor = useCallback(async (stream: MediaStream) => {
    const context = new AudioContext();
    endpointAudioContextRef.current = context;
    if (context.state === "suspended") await context.resume();

    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    const silentOutput = context.createGain();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = 0.2;
    silentOutput.gain.value = 0;
    source.connect(analyser);
    analyser.connect(silentOutput);
    silentOutput.connect(context.destination);

    const samples = new Float32Array(analyser.fftSize);
    const detector = new SpeechEndpointDetector();

    const sampleVoiceActivity = () => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state !== "recording") return;

      analyser.getFloatTimeDomainData(samples);
      let squaredTotal = 0;
      for (let index = 0; index < samples.length; index += 1) {
        squaredTotal += samples[index] * samples[index];
      }
      const rms = Math.sqrt(squaredTotal / samples.length);
      const phase = detector.update(rms, performance.now());

      if (phase === "speaking") signalParticipantActivity();

      if (phase === "grace" && recordingStateRef.current !== "confirming") {
        recordingStateRef.current = "confirming";
        setRecordingState("confirming");
      } else if (phase === "speaking" && recordingStateRef.current === "confirming") {
        recordingStateRef.current = "recording";
        setRecordingState("recording");
      } else if (phase === "ended") {
        endpointFrameRef.current = null;
        recorder.stop();
        return;
      }

      endpointFrameRef.current = requestAnimationFrame(sampleVoiceActivity);
    };

    endpointFrameRef.current = requestAnimationFrame(sampleVoiceActivity);
  }, [signalParticipantActivity]);

  const startRecording = useCallback(async () => {
    if (
      !widgetKey ||
      responseModeRef.current !== "voice" ||
      agentTypingRef.current ||
      wsRef.current?.readyState !== WebSocket.OPEN ||
      recordingStateRef.current !== "idle" ||
      mediaRecorderRef.current?.state === "recording"
    ) return;
    setRecordingError("");
    setChatInput("");

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setRecordingError("Voice recording is not supported in this browser. Please type your response instead.");
      setResponseMode("text");
      return;
    }

    try {
      let stream = recordingStreamRef.current;
      if (!stream || !stream.getAudioTracks().some((track) => track.readyState === "live")) {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
        recordingStreamRef.current = stream;
      }
      const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
        .find((candidate) => MediaRecorder.isTypeSupported(candidate));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;
      recordingChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordingChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        releaseTurnMicrophone();
        recordingStateRef.current = "idle";
        setRecordingState("idle");
        setRecordingError("The recording stopped unexpectedly. Please try again or type instead.");
      };
      recorder.onstop = async () => {
        const chunks = recordingChunksRef.current;
        recordingChunksRef.current = [];
        cleanupTurnRecording();
        if (!chunks.length) {
          recordingStateRef.current = "idle";
          setRecordingState("idle");
          if (finishAfterCurrentAnswerRef.current) {
            finishAfterCurrentAnswerRef.current = false;
            sendParticipantText("", true);
          } else {
            setRecordingError("We did not capture any audio. Please try again or type instead.");
          }
          return;
        }

        recordingStateRef.current = "transcribing";
        setRecordingState("transcribing");
        try {
          const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
          const form = new FormData();
          const extension = blob.type.includes("mp4") ? "m4a" : "webm";
          form.append("audio", blob, `answer.${extension}`);
          const response = await fetch(`/api/widget/${widgetKey}/transcribe`, {
            method: "POST",
            body: form,
          });
          const payload = await response.json().catch(() => ({}));
          if (!response.ok) throw new Error(payload.detail || "We could not transcribe that response.");
          const transcript = String(payload.text || "").trim();
          if (!transcript) throw new Error("We could not hear a response. Please try again or type instead.");
          const shouldFinishInterview = finishAfterCurrentAnswerRef.current;
          finishAfterCurrentAnswerRef.current = false;
          if (!sendParticipantText(transcript, shouldFinishInterview)) {
            setChatInput(transcript);
            responseModeRef.current = "text";
            setResponseMode("text");
            recordingStateRef.current = "idle";
            setRecordingState("idle");
            setRecordingError("Your transcript is ready, but the connection is not. Please send it when the interview reconnects.");
          }
        } catch (error) {
          if (finishAfterCurrentAnswerRef.current) {
            finishAfterCurrentAnswerRef.current = false;
            recordingStateRef.current = "idle";
            setRecordingState("idle");
            setRecordingError("");
            sendParticipantText("", true);
            return;
          }
          recordingStateRef.current = "idle";
          setRecordingState("idle");
          setRecordingError(error instanceof Error ? error.message : "Voice transcription failed. Please try again.");
        }
      };

      recorder.start(250);
      setRecordingState("recording");
      recordingStateRef.current = "recording";
      setRecordingSeconds(0);
      const startedAt = Date.now();
      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds(Math.floor((Date.now() - startedAt) / 1000));
      }, 250);
      try {
        await startEndpointMonitor(stream);
      } catch {
        setRecordingError("Automatic pause detection is unavailable here. Use Send answer now, or switch to typing.");
      }
    } catch (error) {
      releaseTurnMicrophone();
      recordingStateRef.current = "idle";
      const name = error instanceof DOMException ? error.name : "";
      setRecordingError(
        name === "NotAllowedError" || name === "PermissionDeniedError"
          ? "Microphone access was not allowed. You can enable it in your browser or type instead."
          : "We could not start the microphone. Please try again or type instead."
      );
    }
  }, [cleanupTurnRecording, releaseTurnMicrophone, sendParticipantText, startEndpointMonitor, widgetKey]);

  startRecordingRef.current = startRecording;

  const finishRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }, []);

  // ── Send text message ──────────────────────────────────────
  const handleSendMessage = useCallback(() => {
    sendParticipantText(chatInput);
  }, [chatInput, sendParticipantText]);

  const handleStop = useCallback(() => {
    setFinishDialogOpen(true);
  }, []);

  const confirmFinishInterview = useCallback(() => {
    setFinishDialogOpen(false);
    stopPromptAudio();

    // Preserve a spoken answer already in progress: finish capture, transcribe
    // it, and submit it atomically with the session-finish control message.
    if (
      recordingStateRef.current === "recording" ||
      recordingStateRef.current === "confirming" ||
      recordingStateRef.current === "transcribing"
    ) {
      finishAfterCurrentAnswerRef.current = true;
      finishingInterviewRef.current = true;
      setFinishingInterview(true);
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
      }
      return;
    }

    if (isTextChat && sendParticipantText(chatInput, true)) return;

    // Legacy voice transport does not have the text-chat control protocol.
    // Closing it still finalises the server-side session.
    cleanupConnection();
    setEndedReason("user_stopped");
    setStatus("ended");
  }, [chatInput, cleanupConnection, isTextChat, sendParticipantText, stopPromptAudio]);

  const handleAudioToggle = useCallback(() => {
    const next = !audioEnabled;
    audioEnabledRef.current = next;
    setAudioEnabled(next);
    setAudioError("");
    if (!next) {
      stopPromptAudio();
      if (responseModeRef.current === "voice") {
        window.setTimeout(() => void startRecordingRef.current?.(), 0);
      }
      return;
    }
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === "recording") {
      recorder.onstop = null;
      recorder.stop();
      recordingChunksRef.current = [];
      cleanupTurnRecording();
      recordingStateRef.current = "idle";
      setRecordingState("idle");
      setRecordingSeconds(0);
    }
    const latestAgentMessage = [...chatMessages]
      .reverse()
      .find((message) => message.role === "agent");
    if (latestAgentMessage) void playPromptAudio(latestAgentMessage.text, true);
  }, [audioEnabled, chatMessages, cleanupTurnRecording, playPromptAudio, stopPromptAudio]);

  const handleSubmitParticipantId = (e: React.FormEvent) => {
    e.preventDefault();
    if (participantId.trim()) {
      setStatus("idle");
    }
  };

  // ── Format elapsed time ─────────────────────────────────────
  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`;
  };

  const voiceState: "connecting" | "listening" | "speaking" =
    status === "connecting" ? "connecting"
    : agentSpeaking ? "speaking"
    : "listening";
  const latestAgentMessage = [...chatMessages]
    .reverse()
    .find((message) => message.role === "agent");

  // ── Render ──────────────────────────────────────────────────

  if (status === "loading") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="h-8 w-8 rounded-full border-2 border-gray-300 border-t-gray-800 animate-spin" />
          <span className="text-gray-400 text-sm">Loading…</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center p-6 selection:bg-gray-900/10"
      style={{
        background: `radial-gradient(ellipse at 50% 30%, ${hexToRgba(primaryColor, 0.06)} 0%, transparent 70%), linear-gradient(180deg, #fafafa 0%, #f3f4f6 100%)`,
        "--orb-rgb": hexToRgb(primaryColor),
      } as React.CSSProperties}
    >
      <div className={`w-full flex flex-col items-center ${status === "idle" || status === "input" ? "max-w-2xl" : "max-w-md"}`}>
        {/* ── Title & description ─────────────────────────────── */}
        {!immersiveInterview && <div className="text-center mb-10 animate-fade-in">
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight mb-2">
            {config?.widget_title || "Research Interview"}
          </h1>
          {status !== "ended" && config?.widget_description && (
            <p className="text-sm text-gray-500 max-w-sm mx-auto leading-relaxed">
              {config.widget_description}
            </p>
          )}
          {status === "ended" && (
            <div className="animate-fade-in">
              <div className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium ${
                  endedReason === "user_stopped" || endedReason === "inactive"
                    ? "bg-gray-100 text-gray-700"
                    : "bg-emerald-50 text-emerald-700"
                }`}>
                <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                {endedReason === "user_stopped"
                  ? "You ended the interview"
                  : endedReason === "inactive"
                    ? "Interview closed for inactivity"
                    : "Interview complete"}
              </div>
              <p className="text-sm text-gray-400 mt-3">
                {endedReason === "user_stopped" || endedReason === "inactive"
                  ? "Your responses so far have been saved. You may close this page."
                  : "Thank you for your participation!"}
              </p>
              {returnUrl && <div className="mt-5"><p className="mb-3 text-xs text-gray-500">Returning you to Prolific…</p><a href={returnUrl} className="inline-flex rounded-xl bg-gray-900 px-5 py-2.5 text-sm font-semibold text-white">Return to Prolific now</a></div>}
            </div>
          )}
        </div>}

        {/* ── Structured progress bar ────────────────────────── */}
        {participantInfo && (status === "idle" || status === "input") && (
          <section className="mb-8 w-full rounded-2xl border border-gray-200 bg-white p-6 shadow-sm animate-slide-up sm:p-7" aria-labelledby="study-information-heading">
            <div className="flex items-center justify-between gap-3 mb-3">
              <h2 id="study-information-heading" className="text-lg font-semibold text-gray-900">Before you begin</h2>
              {config?.research_stage && (
                <span className="rounded-full bg-gray-100 px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-gray-500">{config.research_stage}</span>
              )}
            </div>
            {participantInfo.purpose_summary && <p className="text-sm text-gray-700 leading-relaxed">{participantInfo.purpose_summary}</p>}

            <div className="mt-6 grid gap-6 text-sm leading-relaxed text-gray-600 sm:grid-cols-2">
              <div>
                <h3 className="font-semibold text-gray-900">What you’ll do</h3>
                <div className="mt-2 space-y-2">
                  {participantInfo.participation_summary && <p>{participantInfo.participation_summary}</p>}
                  {participantInfo.estimated_minutes && <p><span className="font-medium text-gray-700">Time:</span> about {participantInfo.estimated_minutes} minutes. The interviewer will wrap up naturally rather than cut off your current answer; a technical safety limit closes the session a few minutes later.</p>}
                  {participantInfo.ai_disclosure && <p>{participantInfo.ai_disclosure}</p>}
                  <p>You may ask about the study, its scope, or what a question means at any time.</p>
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Privacy and data use</h3>
                <div className="mt-2 space-y-2">
                  {participantInfo.data_handling_summary && <p>{participantInfo.data_handling_summary}</p>}
                  {participantInfo.data_controller && <p><span className="font-medium text-gray-700">Data controller:</span> {participantInfo.data_controller}</p>}
                  {participantInfo.data_recipients_summary && <p><span className="font-medium text-gray-700">Who receives the data:</span> {participantInfo.data_recipients_summary}</p>}
                  {participantInfo.data_retention_period && <p><span className="font-medium text-gray-700">Retention:</span> {participantInfo.data_retention_period}</p>}
                  {participantInfo.privacy_reminder && <p>{participantInfo.privacy_reminder}</p>}
                  {safeHandoffUrl(participantInfo.privacy_notice_url) && <p><a className="font-medium underline" href={safeHandoffUrl(participantInfo.privacy_notice_url) || undefined} target="_blank" rel="noreferrer">Read the full privacy notice</a></p>}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Risks, benefits, and payment</h3>
                <div className="mt-2 space-y-2">
                  {participantInfo.risks_summary && <p><span className="font-medium text-gray-700">Possible risks or discomforts:</span> {participantInfo.risks_summary}</p>}
                  {participantInfo.benefits_summary && <p><span className="font-medium text-gray-700">Benefits:</span> {participantInfo.benefits_summary}</p>}
                  {participantInfo.compensation_summary && <p><span className="font-medium text-gray-700">Compensation:</span> {participantInfo.compensation_summary}</p>}
                </div>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">Your choices and contacts</h3>
                <div className="mt-2 space-y-2">
                  {participantInfo.rights.length > 0 && <ul className="list-disc space-y-1 pl-5">{participantInfo.rights.map((right, index) => <li key={index}>{right}</li>)}</ul>}
                  {participantInfo.withdrawal_instructions && <p><span className="font-medium text-gray-700">Withdrawal or deletion:</span> {participantInfo.withdrawal_instructions}</p>}
                  {participantInfo.researcher_identity && <p><span className="font-medium text-gray-700">Researcher:</span> {participantInfo.researcher_identity}</p>}
                  {participantInfo.researcher_contact && <p><span className="font-medium text-gray-700">Study questions:</span> {participantInfo.researcher_contact}</p>}
                  {participantInfo.complaints_contact && <p><span className="font-medium text-gray-700">Complaints or concerns:</span> {participantInfo.complaints_contact}</p>}
                  {participantInfo.ethics_review_reference && <p><span className="font-medium text-gray-700">Ethics review:</span> {participantInfo.ethics_review_reference}</p>}
                </div>
              </div>
            </div>

            {config?.voice_response_available && (
              <div className="mt-6 rounded-2xl border border-gray-200 bg-gray-50 p-4">
                <h3 className="text-sm font-semibold text-gray-900">Choose how to answer</h3>
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  <button type="button" onClick={() => { responseModeRef.current = "voice"; setResponseMode("voice"); }} className={`rounded-xl border p-3 text-left text-sm ${responseMode === "voice" ? "border-gray-900 bg-white ring-1 ring-gray-900" : "border-gray-200 bg-white text-gray-600"}`}><span className="flex items-center gap-2 font-semibold"><MicIcon className="h-4 w-4" /> Speak your answers</span><span className="mt-1 block text-xs font-normal text-gray-500">Recommended. Your browser will ask for microphone access.</span></button>
                  <button type="button" onClick={() => { cleanupMicCheck(); responseModeRef.current = "text"; setResponseMode("text"); }} className={`rounded-xl border p-3 text-left text-sm ${responseMode === "text" ? "border-gray-900 bg-white ring-1 ring-gray-900" : "border-gray-200 bg-white text-gray-600"}`}><span className="font-semibold">Type your answers</span><span className="mt-1 block text-xs font-normal text-gray-500">No microphone permission is needed.</span></button>
                </div>
                {responseMode === "voice" && <div className="mt-4 text-xs leading-relaxed text-gray-600"><p>Use a quiet place if possible. Check that your microphone is connected and not muted. Questions appear in writing; pause when you finish and your answer will be transcribed and sent automatically. If we stop too soon, keep speaking during the short confirmation period.</p><p className="mt-2">The microphone is used for transcription during each answer. The data-handling statement above governs whether audio or transcripts are retained.</p><div className="mt-3 flex flex-wrap items-center gap-3"><button type="button" onClick={() => void runMicrophoneCheck()} disabled={micCheckStatus === "checking"} className="rounded-lg border border-gray-300 bg-white px-3 py-2 font-semibold text-gray-700 hover:bg-gray-100 disabled:opacity-50">{micCheckStatus === "checking" ? "Checking…" : "Test microphone"}</button>{micCheckMessage && <span role="status" className={micCheckStatus === "ready" ? "text-emerald-700" : micCheckStatus === "denied" || micCheckStatus === "quiet" ? "text-amber-700" : "text-gray-500"}>{micCheckMessage}</span>}</div></div>}
                {config?.audio_available && <p className="mt-3 text-xs text-gray-500">The interviewer’s voice is off by default. You can turn question audio on during the interview.</p>}
              </div>
            )}
            {participantInfo.consent_required && <>
              <label className="mt-6 flex items-start gap-3 rounded-xl border border-gray-200 bg-white px-4 py-4 text-sm font-medium text-gray-700 cursor-pointer">
                <input type="checkbox" checked={consentAccepted} onChange={(e) => setConsentAccepted(e.target.checked)} className="mt-0.5" />
                <span>{participantInfo.consent_label}</span>
              </label>
              <button type="button" onClick={() => { const target = safeHandoffUrl(config?.fielding?.no_consent_url); if (target) window.location.assign(target); else { setErrorMsg("You have not consented. Please return this submission from the recruitment platform."); setStatus("error"); } }} className="mt-3 text-xs font-medium text-gray-500 underline hover:text-gray-800">I do not consent</button>
            </>}
            {safeHandoffUrl(config?.source_code_url) && <p className="mt-5 border-t border-gray-100 pt-4 text-xs text-gray-500">This service uses AGPL-licensed software. <a className="font-medium underline" href={safeHandoffUrl(config?.source_code_url) || undefined} target="_blank" rel="noreferrer">View the source code for this deployment</a>.</p>}
          </section>
        )}

        {/* ── Listen-style text-first interview ──────────────── */}
        {isTextChat && (status === "active" || status === "connecting") && (
          <section className="fixed inset-0 z-20 flex min-h-screen flex-col bg-[#fbfbfa] animate-fade-in" aria-live="polite">
            <header className="flex h-16 shrink-0 items-center justify-between border-b border-gray-200/80 bg-white/90 px-4 backdrop-blur sm:px-7">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-gray-900">{config?.widget_title || "Research interview"}</p>
                <p className="mt-0.5 truncate text-[10px] font-medium uppercase tracking-[0.16em] text-gray-400">AI-moderated interview</p>
              </div>
              <div className="ml-4 flex items-center gap-2 sm:gap-3">
                {config?.audio_available && (
                  <button
                    type="button"
                    role="switch"
                    aria-checked={audioEnabled}
                    aria-label={audioEnabled ? "Turn interview audio off" : "Turn interview audio on"}
                    onClick={handleAudioToggle}
                    className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition ${audioEnabled ? "border-gray-900 bg-gray-900 text-white" : "border-gray-200 bg-white text-gray-600 hover:border-gray-300"}`}
                  >
                    {audioLoading ? <Spinner className="h-3.5 w-3.5" /> : <SpeakerIcon className="h-3.5 w-3.5" muted={!audioEnabled} />}
                    <span className="hidden sm:inline">Audio {audioEnabled ? "on" : "off"}</span>
                  </button>
                )}
                <span className="hidden font-mono text-xs tabular-nums text-gray-400 sm:block">{formatTime(elapsed)}</span>
                <button onClick={handleStop} disabled={finishingInterview} className="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 hover:border-red-200 hover:bg-red-50 hover:text-red-600 disabled:cursor-wait disabled:opacity-50">{finishingInterview ? "Finishing…" : "Finish interview"}</button>
              </div>
            </header>

            {showProgress && (
              <div className="shrink-0 border-b border-gray-200 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(0,0,0,0.03)] sm:px-7">
                <div className="mb-2 flex items-center justify-between gap-4">
                  <span className="text-xs font-medium text-gray-600">Interview progress</span>
                  <span className="text-xs font-semibold text-gray-900">{approximateProgressLabel}</span>
                </div>
                <div
                  className="h-2 w-full overflow-hidden rounded-full bg-gray-200"
                  role="progressbar"
                  aria-label="Approximate interview progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={approximateProgressPercent}
                  aria-valuetext={approximateProgressLabel}
                >
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: `${approximateProgressPercent}%`,
                      backgroundColor: primaryColor,
                    }}
                  />
                </div>
              </div>
            )}

            {inactivityWarning && (
              <div className="shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-3 sm:px-7" role="alert">
                <div className="mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-amber-950">Still there?</p>
                    <p className="mt-0.5 text-xs text-amber-800">
                      Listening is paused. This interview will close and save in {inactivitySecondsRemaining} second{inactivitySecondsRemaining === 1 ? "" : "s"} unless you continue.
                    </p>
                  </div>
                  <button type="button" onClick={resumeAfterInactivity} className="rounded-lg bg-amber-900 px-4 py-2 text-xs font-semibold text-white hover:bg-amber-950">
                    I’m still here
                  </button>
                </div>
              </div>
            )}

            <main id="chat-scroll" className="flex-1 overflow-y-auto">
              <div className="mx-auto w-full max-w-3xl px-5 pb-48 pt-10 sm:px-8 sm:pb-52 sm:pt-14">
                {status === "connecting" && chatMessages.length === 0 && (
                  <div className="flex min-h-[45vh] items-center justify-center text-sm text-gray-400"><Spinner className="mr-2 h-4 w-4" />Connecting to your interviewer…</div>
                )}

                <div className="space-y-10 sm:space-y-12">
                  {chatMessages.map((msg) => {
                    const isLatestPrompt = msg.role === "agent" && msg.id === latestAgentMessage?.id;
                    return (
                      <article ref={isLatestPrompt ? latestPromptRef : undefined} key={msg.id} className="animate-fade-in scroll-mt-10">
                        <div className="mb-3 flex items-center justify-between gap-3">
                          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-400">{msg.role === "agent" ? "Interviewer" : "You"}</span>
                          {isLatestPrompt && config?.audio_available && (
                            <button type="button" onClick={() => void playPromptAudio(msg.text, true)} className="flex items-center gap-1.5 text-[11px] font-medium text-gray-400 hover:text-gray-700" aria-label="Read this question aloud">
                              <SpeakerIcon className="h-3.5 w-3.5" /> Read aloud
                            </button>
                          )}
                        </div>
                        {msg.role === "agent" ? (
                          <div className={`${isLatestPrompt ? "text-[26px] leading-[1.3] tracking-[-0.025em] text-gray-950 sm:text-[30px]" : "text-lg leading-relaxed text-gray-500"} font-medium`}>
                            <ChatBubbleContent text={msg.text} isUser={false} />
                          </div>
                        ) : (
                          <div className="rounded-2xl border border-gray-200 bg-white px-5 py-4 text-base leading-relaxed text-gray-700 shadow-sm sm:px-6">
                            <ChatBubbleContent text={msg.text} isUser />
                          </div>
                        )}
                      </article>
                    );
                  })}

                  {agentTyping && (
                    <div className="flex items-center gap-2 text-xs text-gray-400 animate-fade-in">
                      <span className="typing-dot" style={{ animationDelay: "0s" }} />
                      <span className="typing-dot" style={{ animationDelay: "0.15s" }} />
                      <span className="typing-dot" style={{ animationDelay: "0.3s" }} />
                      <span className="ml-1">Interviewer is thinking…</span>
                    </div>
                  )}
                </div>
              </div>
            </main>

            <footer className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#fbfbfa] via-[#fbfbfa] to-transparent px-4 pb-4 pt-10 sm:px-6 sm:pb-6">
              <div className="mx-auto w-full max-w-3xl">
                {config?.voice_response_available && (
                  <div className="mb-2 flex justify-center">
                    <div className="inline-flex rounded-full border border-gray-200 bg-white p-1 text-xs shadow-sm" role="group" aria-label="Response method">
                      <button
                        type="button"
                        onClick={() => {
                          responseModeRef.current = "voice";
                          setResponseMode("voice");
                          window.setTimeout(() => void startRecordingRef.current?.(), 0);
                        }}
                        disabled={recordingState === "transcribing"}
                        className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 font-medium transition ${responseMode === "voice" ? "bg-gray-900 text-white" : "text-gray-500 hover:text-gray-800"}`}
                      >
                        <MicIcon className="h-3.5 w-3.5" /> Speak
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          responseModeRef.current = "text";
                          cancelRecording();
                          setResponseMode("text");
                        }}
                        disabled={recordingState === "transcribing"}
                        className={`rounded-full px-3 py-1.5 font-medium transition ${responseMode === "text" ? "bg-gray-900 text-white" : "text-gray-500 hover:text-gray-800"}`}
                      >
                        Type
                      </button>
                    </div>
                  </div>
                )}

                {responseMode === "voice" && config?.voice_response_available ? (
                  <div className="rounded-2xl border border-gray-200 bg-white p-3 shadow-[0_18px_55px_rgba(17,24,39,0.12)]">
                    {recordingState === "idle" && (
                      <button
                        type="button"
                        onClick={() => void startRecording()}
                        disabled={status !== "active" || agentTyping}
                        className="flex min-h-16 w-full items-center justify-center gap-3 rounded-xl font-medium text-white transition hover:opacity-90 active:scale-[0.995] disabled:cursor-not-allowed disabled:opacity-40"
                        style={{ backgroundColor: primaryColor }}
                      >
                        <span className="flex h-10 w-10 items-center justify-center rounded-full bg-white/15"><MicIcon className="h-5 w-5" /></span>
                        <span>{agentTyping ? "The interviewer is responding…" : "Tap to answer by voice"}</span>
                      </button>
                    )}

                    {(recordingState === "recording" || recordingState === "confirming") && (
                      <div className="flex min-h-16 items-center justify-between gap-4 px-2">
                        <div className="flex items-center gap-3">
                          <span className="relative flex h-3 w-3"><span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${recordingState === "confirming" ? "bg-amber-400" : "bg-red-400"}`} /><span className={`relative inline-flex h-3 w-3 rounded-full ${recordingState === "confirming" ? "bg-amber-500" : "bg-red-500"}`} /></span>
                          <div>
                            <p className="text-sm font-medium text-gray-900">{recordingState === "confirming" ? "Finishing your answer…" : "Listening…"}</p>
                            <p className="mt-0.5 text-xs text-gray-400">{recordingState === "confirming" ? "Keep speaking if you’re not done" : `${formatTime(recordingSeconds)} · we’ll wait through a natural pause`}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <button type="button" onClick={cancelRecording} className="rounded-lg px-3 py-2 text-xs font-medium text-gray-500 hover:bg-gray-100">Cancel</button>
                          <button type="button" onClick={finishRecording} className="rounded-xl bg-gray-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-gray-800" title="Ends this answer, not the interview">Send answer now</button>
                        </div>
                      </div>
                    )}

                    {recordingState === "transcribing" && (
                      <div className="flex min-h-16 items-center justify-center gap-3 text-sm font-medium text-gray-600"><Spinner className="h-4 w-4" />Transcribing and sending…</div>
                    )}
                  </div>
                ) : (
                  <form
                    onSubmit={(event) => { event.preventDefault(); handleSendMessage(); }}
                    className="rounded-2xl border border-gray-200 bg-white p-2 shadow-[0_18px_55px_rgba(17,24,39,0.12)]"
                  >
                    <div className="flex items-end gap-2">
                      <textarea
                        ref={chatInputRef}
                        rows={1}
                        value={chatInput}
                        onChange={(event) => { setChatInput(event.target.value); resizeChatInput(event.target); signalParticipantActivity(); }}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" && !event.shiftKey) {
                            event.preventDefault();
                            handleSendMessage();
                          }
                        }}
                        className="min-h-14 flex-1 resize-none overflow-y-auto border-0 bg-transparent px-3 py-3 text-base leading-relaxed text-gray-900 outline-none placeholder:text-gray-400 focus:ring-0"
                        placeholder={agentTyping ? "The interviewer is responding…" : "Write your response…"}
                        aria-label="Your response"
                        autoFocus
                        disabled={status !== "active" || agentTyping}
                      />
                      <button type="submit" aria-label="Send response" disabled={!chatInput.trim() || status !== "active" || agentTyping} className="mb-1 flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-white hover:opacity-90 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30" style={{ backgroundColor: primaryColor }}><SendIcon className="h-5 w-5" /></button>
                    </div>
                    <div className="flex items-center justify-between gap-3 px-3 pb-1 text-[10px] text-gray-400"><span>Typing mode</span><span className="hidden sm:inline">Enter to send · Shift+Enter for a new line</span></div>
                  </form>
                )}
                {(recordingError || audioError) && <p role="status" className="mt-2 text-center text-[11px] text-amber-700">{recordingError || audioError}</p>}
              </div>
            </footer>
          </section>
        )}

        {/* ── Text-first spoken interview ───────────────────── */}
        {!isTextChat && (status === "active" || status === "connecting") && (
          <section className="fixed inset-0 z-10 flex min-h-screen flex-col bg-[#fdfdfd] animate-fade-in" aria-live="polite">
            <div className="h-1 w-full bg-gray-100">
              <div
                className="h-full transition-all duration-500"
                style={{
                  width: showProgress && progress
                    ? `${Math.min(100, Math.max(3, (progress.current / Math.max(1, progress.total)) * 100))}%`
                    : status === "connecting" ? "4%" : "8%",
                  backgroundColor: primaryColor,
                }}
              />
            </div>
            <header className="flex items-center justify-between px-5 py-4 sm:px-8">
              <div className="flex items-center gap-5 text-xs text-gray-500">
                <span>English <span aria-hidden>⌄</span></span>
                <span className="flex items-center gap-1.5"><SoundWaveIcon className="h-4 w-4" /> Read aloud</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs tabular-nums text-gray-400">{formatTime(elapsed)}</span>
                <button onClick={handleStop} disabled={finishingInterview} className="rounded-full border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:border-red-200 hover:bg-red-50 hover:text-red-600 disabled:cursor-wait disabled:opacity-50">{finishingInterview ? "Finishing…" : "Finish interview"}</button>
              </div>
            </header>

            <div className="flex flex-1 flex-col items-center justify-center px-6 pb-20 pt-8 text-center sm:px-12">
              {showProgress && (
                <p className="mb-5 text-[11px] font-semibold uppercase tracking-[0.2em] text-gray-400">
                  {progress ? `Question ${Math.max(1, progress.current)} of ${progress.total}` : "Starting interview"}
                </p>
              )}
              <h1 className="max-w-4xl text-balance text-3xl font-medium leading-[1.25] tracking-[-0.025em] text-gray-900 sm:text-4xl lg:text-[44px]">
                {voicePrompt || (status === "connecting" ? "Connecting to your interviewer…" : config?.welcome_message)}
              </h1>
              <div className="mt-10 flex min-h-10 items-center justify-center">
                {voiceState === "connecting" ? (
                  <span className="flex items-center gap-2 text-xs text-gray-400"><Spinner className="h-4 w-4" /> Connecting…</span>
                ) : voiceState === "speaking" ? (
                  <span className="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-4 py-2 text-xs font-medium text-gray-600 shadow-sm"><SoundWaveIcon className="h-4 w-4" /><span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-40" style={{ backgroundColor: primaryColor }} /><span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: primaryColor }} /></span>Interviewer is speaking</span>
                ) : userSpeaking ? (
                  <span className="flex items-center gap-3 rounded-full border border-gray-200 bg-white px-4 py-2 text-xs font-medium text-gray-600 shadow-sm"><VoiceWaveBars level={audioLevel} color={primaryColor} />You are speaking</span>
                ) : (
                  <span className="flex items-center gap-2 rounded-full border border-gray-200 bg-white px-4 py-2 text-xs font-medium text-gray-600 shadow-sm"><span className="h-2 w-2 rounded-full bg-emerald-500" />{listeningMessage}</span>
                )}
              </div>
              {voiceUserText && !agentSpeaking && (
                <details className="mt-8 max-w-2xl text-left">
                  <summary className="cursor-pointer text-center text-xs text-gray-400 hover:text-gray-600">Review what I just said</summary>
                  <p className="mt-2 rounded-xl bg-gray-50 px-4 py-3 text-sm leading-relaxed text-gray-600">{voiceUserText}</p>
                </details>
              )}
            </div>
            <footer className="px-5 pb-6 text-center text-[11px] text-gray-400">Speak naturally. You can pause, skip a question, or end at any time.</footer>
          </section>
        )}

        {/* ── Idle orb/icon (before starting) ─────────────────── */}
        {status === "idle" && isTextChat && (
          <div className="relative mb-10 flex items-center justify-center animate-scale-in">
            <div
              className="absolute rounded-full animate-orb-breathe"
              style={{
                width: 140,
                height: 140,
                backgroundColor: hexToRgba(primaryColor, 0.06),
              }}
            />
            <div
              className="rounded-full flex items-center justify-center"
              style={{
                width: 120,
                height: 120,
                background: `radial-gradient(circle at 35% 30%, ${hexToRgba(primaryColor, 0.75)}, ${hexToRgba(primaryColor, 0.55)} 80%)`,
                boxShadow: `0 0 0 1px ${hexToRgba(primaryColor, 0.08)}, 0 8px 24px ${hexToRgba(primaryColor, 0.12)}`,
              }}
            >
              {config?.voice_response_available ? <MicIcon className="w-8 h-8 text-white opacity-80" /> : <ChatIcon className="w-8 h-8 text-white opacity-80" />}
            </div>
          </div>
        )}
        {status === "idle" && !isTextChat && (
          <div className="relative mb-10 flex items-center justify-center animate-scale-in">
            {/* Breathing glow */}
            <div
              className="absolute rounded-full animate-orb-breathe"
              style={{
                width: 140,
                height: 140,
                backgroundColor: hexToRgba(primaryColor, 0.06),
              }}
            />
            <div
              className="rounded-full flex items-center justify-center"
              style={{
                width: 120,
                height: 120,
                background: `radial-gradient(circle at 35% 30%, ${hexToRgba(primaryColor, 0.75)}, ${hexToRgba(primaryColor, 0.55)} 80%)`,
                boxShadow: `0 0 0 1px ${hexToRgba(primaryColor, 0.08)}, 0 8px 24px ${hexToRgba(primaryColor, 0.12)}`,
              }}
            >
              {/* Glass highlight */}
              <div
                className="absolute rounded-full"
                style={{
                  top: "calc(50% - 60px + 14%)",
                  left: "calc(50% - 60px + 20%)",
                  width: "36px",
                  height: "24px",
                  background: "linear-gradient(180deg, rgba(255,255,255,0.3) 0%, transparent 100%)",
                  borderRadius: "50%",
                  filter: "blur(4px)",
                }}
              />
              <MicIcon className="w-7 h-7 text-white opacity-70" />
            </div>
          </div>
        )}

        {/* ── Error message ───────────────────────────────────── */}
        {status === "error" && (
          <div className="mb-6 w-full animate-slide-up">
            <div className="rounded-2xl bg-red-50 border border-red-100 px-5 py-4 text-center">
              <p className="text-sm text-red-700">{errorMsg}</p>
            </div>
          </div>
        )}

        {/* ── Participant ID input (input mode) ───────────────── */}
        {status === "input" && (
          <form
            onSubmit={handleSubmitParticipantId}
            className="mb-6 w-full rounded-2xl border border-gray-200 bg-white p-6 shadow-sm animate-slide-up"
          >
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Enter your Participant ID
            </label>
            <p className="text-xs text-gray-400 mb-4">
              Please enter the identifier provided to you by the researcher.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={participantId}
                onChange={(e) => setParticipantId(e.target.value)}
                className="input-styled flex-1"
                placeholder="e.g. P001"
                autoFocus
                required
              />
              <button
                type="submit"
                className="rounded-xl px-5 py-2.5 text-sm font-medium text-white transition-all hover:opacity-90 active:scale-[0.98]"
                style={{ backgroundColor: primaryColor }}
              >
                Continue
              </button>
            </div>
          </form>
        )}

        {/* ── Action buttons ──────────────────────────────────── */}
        {!immersiveInterview && <div className="flex justify-center animate-fade-in">
          {status === "idle" && (
            <button
              onClick={isTextChat ? handleStartChat : handleStart}
              disabled={!canStart}
              className="group flex items-center gap-3 rounded-2xl px-8 py-4 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-[1.03] active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
              style={{
                backgroundColor: primaryColor,
                boxShadow: `0 8px 30px ${hexToRgba(primaryColor, 0.25)}`,
              }}
            >
              {config?.voice_response_available ? <MicIcon className="w-5 h-5" /> : <ChatIcon className="w-5 h-5" />}
              Start Interview
            </button>
          )}

          {status === "ended" && (
            <p className="text-gray-400 text-sm animate-fade-in">
              You may close this page now.
            </p>
          )}

          {status === "error" && (
            <button
              onClick={() => {
                cleanupConnection();
                setStatus(
                  config?.participant_id_mode === "input" ? "input" : "idle"
                );
                setErrorMsg("");
              }}
              className="rounded-2xl px-8 py-4 text-white font-semibold shadow-lg transition-all duration-300 hover:scale-[1.03] active:scale-[0.98]"
              style={{
                backgroundColor: primaryColor,
                boxShadow: `0 8px 30px ${hexToRgba(primaryColor, 0.25)}`,
              }}
            >
              Try Again
            </button>
          )}
        </div>}
      </div>

      {/* Branding footer */}
      {!immersiveInterview && <p className="mt-16 text-[11px] text-gray-300 tracking-wider uppercase">
        Powered by OASIS
      </p>}

      {finishDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/35 px-5 backdrop-blur-[2px]" role="presentation">
          <section role="dialog" aria-modal="true" aria-labelledby="finish-interview-title" className="w-full max-w-md rounded-3xl border border-gray-200 bg-white p-6 shadow-2xl sm:p-7">
            <h2 id="finish-interview-title" className="text-xl font-semibold tracking-tight text-gray-950">Finish this interview?</h2>
            <p className="mt-3 text-sm leading-relaxed text-gray-600">
              Your responses so far will be submitted. You can finish before the suggested time, and any answer you are currently recording will be transcribed and included.
            </p>
            <div className="mt-6 flex justify-end gap-3">
              <button type="button" onClick={() => setFinishDialogOpen(false)} className="rounded-xl px-4 py-2.5 text-sm font-medium text-gray-600 hover:bg-gray-100">Keep going</button>
              <button type="button" onClick={confirmFinishInterview} className="rounded-xl bg-gray-950 px-4 py-2.5 text-sm font-medium text-white hover:bg-gray-800">Finish interview</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

// ── Voice Wave Bars (user speaking visualisation) ────────────────

function VoiceWaveBars({ level, color }: { level: number; color: string }) {
  const barCount = 5;
  const bars = [];
  for (let i = 0; i < barCount; i++) {
    // Create varied heights based on position and audio level
    const centerDistance = Math.abs(i - (barCount - 1) / 2);
    const baseHeight = 1 - centerDistance * 0.3;
    const speed = 0.4 + Math.random() * 0.3;
    const delay = i * 0.08;

    bars.push(
      <div
        key={i}
        className="voice-bar rounded-full"
        style={{
          width: 4,
          height: 24 * baseHeight * Math.max(0.3, level),
          backgroundColor: color,
          opacity: 0.85,
          "--bar-speed": `${speed}s`,
          "--bar-delay": `${delay}s`,
        } as React.CSSProperties}
      />
    );
  }

  return (
    <div className="flex items-end gap-[3px]" style={{ height: 24 }}>
      {bars}
    </div>
  );
}

// ── SVG Icons ──────────────────────────────────────────────────

function MicIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.8}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M19 10v2a7 7 0 01-14 0v-2"
      />
      <line x1="12" y1="19" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

function Spinner({ className }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} fill="none" viewBox="0 0 24 24">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  );
}

function ChatIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
      />
    </svg>
  );
}

function SpeakerIcon({ className, muted = false }: { className?: string; muted?: boolean }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M11 5 6.5 9H3v6h3.5L11 19V5Z" />
      {!muted && <path strokeLinecap="round" strokeLinejoin="round" d="M15 9a4 4 0 0 1 0 6m2.5-8.5a7.5 7.5 0 0 1 0 11" />}
      {muted && <path strokeLinecap="round" strokeLinejoin="round" d="m15.5 10 5 5m0-5-5 5" />}
    </svg>
  );
}

function SendIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="m5 12 14-7-4 14-3-5-7-2Z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="m12 14 7-9" />
    </svg>
  );
}

function SoundWaveIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="currentColor" viewBox="0 0 24 24">
      <rect x="4" y="9" width="2" height="6" rx="1">
        <animate attributeName="height" values="6;12;6" dur="0.8s" repeatCount="indefinite" />
        <animate attributeName="y" values="9;6;9" dur="0.8s" repeatCount="indefinite" />
      </rect>
      <rect x="8" y="7" width="2" height="10" rx="1">
        <animate attributeName="height" values="10;4;10" dur="0.6s" repeatCount="indefinite" />
        <animate attributeName="y" values="7;10;7" dur="0.6s" repeatCount="indefinite" />
      </rect>
      <rect x="12" y="5" width="2" height="14" rx="1">
        <animate attributeName="height" values="14;8;14" dur="0.7s" repeatCount="indefinite" />
        <animate attributeName="y" values="5;8;5" dur="0.7s" repeatCount="indefinite" />
      </rect>
      <rect x="16" y="7" width="2" height="10" rx="1">
        <animate attributeName="height" values="10;6;10" dur="0.65s" repeatCount="indefinite" />
        <animate attributeName="y" values="7;9;7" dur="0.65s" repeatCount="indefinite" />
      </rect>
      <rect x="20" y="9" width="2" height="6" rx="1">
        <animate attributeName="height" values="6;10;6" dur="0.75s" repeatCount="indefinite" />
        <animate attributeName="y" values="9;7;9" dur="0.75s" repeatCount="indefinite" />
      </rect>
    </svg>
  );
}
