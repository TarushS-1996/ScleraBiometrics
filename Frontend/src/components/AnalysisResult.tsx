import { useState } from 'react';
import type { IdentifyResponse, SegmentResponse } from '../types';
import { LayoutDashboard, ChevronLeft, ChevronRight } from 'lucide-react';

interface AnalysisResultProps {
    result: IdentifyResponse | SegmentResponse | null;
    allResults?: SegmentResponse[];
    mode: 'verify' | 'add';
    originalImages?: string[];
}

export default function AnalysisResult({ result, allResults = [], mode, originalImages = [] }: AnalysisResultProps) {
    const [currentIndex, setCurrentIndex] = useState(0);

    const [prevResultsLength, setPrevResultsLength] = useState(allResults.length);

    if (allResults.length !== prevResultsLength) {
        setPrevResultsLength(allResults.length);
        setCurrentIndex(0);
    }

    const displayResult = mode === 'add' && allResults.length > 0 ? allResults[currentIndex] : result;
    const isSegment = displayResult && 'processed_image' in displayResult;
    const identifyResult = !isSegment ? displayResult as IdentifyResponse : null;
    const segmentResult = isSegment ? displayResult as SegmentResponse : null;

    const similarity = identifyResult?.best_match?.similarity ?? 0;
    const isMatched = similarity >= 0.7;
    const matchPct = Math.round(similarity * 100);

    // Circle math
    const radius = 45;
    const strokeWidth = 8;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (matchPct / 100) * circumference;

    const nextResult = () => {
        if (allResults.length > 0) {
            setCurrentIndex((prev) => (prev + 1) % allResults.length);
        }
    };

    const prevResult = () => {
        if (allResults.length > 0) {
            setCurrentIndex((prev) => (prev - 1 + allResults.length) % allResults.length);
        }
    };

    const currentOriginal = mode === 'add' ? originalImages[currentIndex] : (originalImages.length > 0 ? originalImages[0] : null);

    return (
        <div className="bg-bg-primary border border-border-custom/50 rounded-xl p-6 flex flex-col h-full shadow-2xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-2.5">
                    <div className="bg-accent-cyan/10 p-1.5 rounded-md">
                        <LayoutDashboard className="w-[18px] h-[18px] text-accent-cyan" />
                    </div>
                    <h3 className="text-[15px] font-bold text-white tracking-wide">
                        {mode === 'verify' ? 'Analysis Result' : 'Images of Eye Captured'}
                    </h3>
                </div>

                {mode === 'add' && allResults.length > 1 && (
                    <div className="flex items-center gap-2">
                        <button
                            onClick={prevResult}
                            className="p-1.5 rounded-md bg-[#151515] border border-[#2a2a2a] text-text-secondary hover:text-white hover:bg-accent-blue/20 transition-all"
                        >
                            <ChevronLeft className="w-4 h-4" />
                        </button>
                        <button
                            onClick={nextResult}
                            className="p-1.5 rounded-md bg-[#151515] border border-[#2a2a2a] text-text-secondary hover:text-white hover:bg-accent-blue/20 transition-all"
                        >
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                )}
            </div>

            <div className="flex-1 flex gap-5 flex-wrap lg:flex-nowrap">
                {!displayResult ? (
                    <div className="flex-1 flex flex-col items-center justify-center text-text-muted border border-border-custom/30 border-dashed rounded-xl min-h-[160px] bg-bg-secondary/30">
                        <div className="w-10 h-10 border-2 border-accent-cyan/20 border-t-accent-cyan rounded-full animate-spin mb-3"></div>
                        <span className="text-[13px] font-medium">Waiting for identification signal...</span>
                    </div>
                ) : isSegment ? (
                    <>
                        {/* Original Image */}
                        <div className="flex-1 min-w-[200px] flex flex-col gap-2">
                            <span className="text-[10px] font-bold text-text-muted uppercase tracking-wider px-1">Original Capture</span>
                            <div className="flex-1 aspect-square bg-black border border-border-custom/50 rounded-xl overflow-hidden group relative">
                                {currentOriginal ? (
                                    <img
                                        src={currentOriginal}
                                        alt="Original"
                                        className="w-full h-full object-cover transition-transform duration-500"
                                    />
                                ) : (
                                    <div className="w-full h-full flex items-center justify-center text-text-muted text-[11px]">No Original</div>
                                )}
                            </div>
                        </div>

                        {/* Processed Image */}
                        <div className="flex-1 min-w-[200px] flex flex-col gap-2">
                            <span className="text-[10px] font-bold text-accent-cyan uppercase tracking-wider px-1">Processed Pattern</span>
                            <div className="flex-1 aspect-square bg-black border border-accent-cyan/50 rounded-xl overflow-hidden shadow-[0_0_15px_rgba(0,191,255,0.15)] group relative">
                                <img
                                    src={`data:image/png;base64,${segmentResult?.processed_image}`}
                                    alt="Processed"
                                    className="w-full h-full object-cover transition-transform duration-500 "
                                />
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        {/* 1. Progress Box */}
                        <div className="flex-1 min-w-[180px] bg-bg-secondary/40 border border-accent-cyan/30 rounded-xl p-5 flex items-center justify-center relative group">
                            <div className="absolute inset-0 bg-accent-cyan/5 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl"></div>
                            <div className="relative w-32 h-32">
                                {/* Background Circle */}
                                <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
                                    <circle
                                        cx="50"
                                        cy="50"
                                        r={radius}
                                        fill="none"
                                        className="stroke-accent-cyan/10"
                                        strokeWidth="1"
                                    />
                                    {/* Outer accent ring */}
                                    <circle
                                        cx="50"
                                        cy="50"
                                        r={radius + 4}
                                        fill="none"
                                        className="stroke-accent-cyan/20"
                                        strokeWidth="1"
                                    />
                                    {/* Main Progress (Green) */}
                                    <circle
                                        cx="50"
                                        cy="50"
                                        r={radius}
                                        fill="none"
                                        className="stroke-accent-green transition-[stroke-dashoffset] duration-1000 ease-out"
                                        strokeWidth={strokeWidth}
                                        strokeLinecap="round"
                                        strokeDasharray={circumference}
                                        strokeDashoffset={offset}
                                    />
                                </svg>
                                {/* Text Overlay */}
                                <div className="absolute inset-0 flex flex-col items-center justify-center pt-1">
                                    <span className="text-2xl font-black text-white leading-none">{matchPct}%</span>
                                    <span className="text-[10px] font-bold text-text-muted tracking-[0.2em] mt-1 ml-1">MATCH</span>
                                </div>
                            </div>
                        </div>

                        {/* 2. Image Box */}
                        <div className="flex-1 min-w-[200px] aspect-square lg:aspect-auto bg-black border border-accent-cyan/50 rounded-xl overflow-hidden shadow-[0_0_15px_rgba(0,191,255,0.15)] group relative">
                            <div className="absolute inset-0 border border-accent-cyan/20 pointer-events-none rounded-xl z-10"></div>
                            {identifyResult?.processed_query_image ? (
                                <img
                                    src={`data:image/png;base64,${identifyResult.processed_query_image}`}
                                    alt="Iris Pattern"
                                    className="w-full h-full object-cover transition-transform duration-500"
                                />
                            ) : (
                                <div className="w-full h-full flex items-center justify-center text-text-muted text-[11px] uppercase tracking-widest text-center px-4 font-mono">
                                    Pattern Not Detected
                                </div>
                            )}
                        </div>

                        {/* 3. Info Box */}
                        <div className="flex-[1.5] min-w-[260px] bg-bg-secondary/40 border border-accent-cyan/30 rounded-xl p-5 flex flex-col gap-4 relative">
                            <div className="flex flex-col gap-1">
                                <h4 className="text-[11px] font-bold text-text-muted uppercase tracking-wider">Final Verification:</h4>
                                <div className="h-0.5 w-8 bg-accent-cyan/30 rounded-full"></div>
                            </div>

                            <div className="grid gap-3">
                                <div className="flex items-center justify-between text-[13px]">
                                    <span className="text-accent-cyan font-medium">Authentication:</span>
                                    <span className={`font-bold tracking-tight ${isMatched ? 'text-accent-green' : 'text-accent-magenta animate-pulse'}`}>
                                        {isMatched ? 'ACCESS GRANTED' : 'ACCESS DENIED'}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[13px]">
                                    <span className="text-accent-cyan font-medium">Liveness:</span>
                                    <span className={`font-medium ${isMatched ? 'text-accent-green' : 'text-text-muted'}`}>
                                        {isMatched ? 'Verified' : 'Not Verified'}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[13px] border-t border-border-custom/20 pt-2">
                                    <span className="text-accent-cyan font-medium">User Name:</span>
                                    <span className="text-white font-semibold truncate max-w-[120px]">
                                        {isMatched ? identifyResult?.best_match.user_id.replace(/_/g, ' ') : 'Not Found'}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[13px]">
                                    <span className="text-accent-cyan font-medium">User ID:</span>
                                    <span className="text-white font-mono text-[12px]">
                                        {isMatched ? identifyResult?.best_match.user_id : '--'}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[13px]">
                                    <span className="text-accent-cyan font-medium">Matched Eye:</span>
                                    <span className="text-white capitalize">
                                        {isMatched ? identifyResult?.best_match.eye_side : '--'}
                                    </span>
                                </div>

                                <div className="flex items-center justify-between text-[13px]">
                                    <span className="text-accent-cyan font-medium">Sample:</span>
                                    <span className="text-text-muted text-[11px] italic truncate max-w-[140px]">
                                        {isMatched ? identifyResult?.best_match.sample : '--'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
