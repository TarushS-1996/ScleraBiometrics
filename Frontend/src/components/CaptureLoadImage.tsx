import { ChevronLeft, ChevronRight, X } from 'lucide-react';

interface CaptureLoadImageProps {
    capturedImages: string[];
    isScanning: boolean;
    currentIndex: number;
    onIndexChange: (index: number) => void;
    onRemoveImage: (index: number) => void;
}

export default function CaptureLoadImage({ 
    capturedImages, 
    isScanning, 
    currentIndex, 
    onIndexChange,
    onRemoveImage 
}: CaptureLoadImageProps) {
    const nextImage = () => {
        if (capturedImages.length > 0) {
            onIndexChange((currentIndex + 1) % capturedImages.length);
        }
    };

    const prevImage = () => {
        if (capturedImages.length > 0) {
            onIndexChange((currentIndex - 1 + capturedImages.length) % capturedImages.length);
        }
    };

    return (
        <div className="bg-bg-secondary border border-border-custom rounded-xl p-4 flex flex-col overflow-hidden min-h-[320px]">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <svg className="w-[18px] h-[18px] text-accent-cyan shrink-0" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" />
                    </svg>
                    <h3 className="text-sm font-semibold">Captured / Loaded image</h3>
                </div>
                
                {capturedImages.length > 1 && (
                    <div className="flex items-center gap-2">
                        <button 
                            onClick={prevImage}
                            className="p-1 rounded bg-[#151515] border border-[#2a2a2a] text-text-secondary hover:text-white hover:bg-accent-blue/20 transition-all"
                        >
                            <ChevronLeft className="w-4 h-4" />
                        </button>
                        <button 
                            onClick={nextImage}
                            className="p-1 rounded bg-[#151515] border border-[#2a2a2a] text-text-secondary hover:text-white hover:bg-accent-blue/20 transition-all"
                        >
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                )}
            </div>

            <div className="flex-1 relative bg-bg-dark rounded-lg overflow-hidden flex items-center justify-center mb-3">
                {capturedImages.length > 0 ? (
                    <img
                        src={capturedImages[currentIndex]}
                        alt={`Captured ${currentIndex + 1}`}
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <div className="flex flex-col items-center justify-center gap-2 text-text-muted text-[13px]">
                        <svg className="w-[42px] h-[42px] opacity-40" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zm-5.04-6.71l-2.75 3.54-1.96-2.36L6.5 17h11l-3.54-4.71z" />
                        </svg>
                        <span>No image captured or loaded</span>
                    </div>
                )}

                {isScanning && (
                    <div className="absolute inset-0 pointer-events-none">
                        <div className="absolute left-[5%] right-[5%] h-0.5 bg-accent-cyan shadow-[0_0_8px_var(--color-accent-cyan),0_0_20px_var(--color-accent-cyan)] animate-[scanMove_3s_ease-in-out_infinite]"></div>
                    </div>
                )}

                <div className="absolute w-6 h-6 border-2 border-accent-cyan top-[5%] left-[5%] border-r-0 border-b-0"></div>
                <div className="absolute w-6 h-6 border-2 border-accent-cyan top-[5%] right-[5%] border-l-0 border-b-0"></div>
                <div className="absolute w-6 h-6 border-2 border-accent-cyan bottom-[5%] left-[5%] border-r-0 border-t-0"></div>
                <div className="absolute w-6 h-6 border-2 border-accent-cyan bottom-[5%] right-[5%] border-l-0 border-t-0"></div>
            </div>

            {/* Clicked Images Icons/Thumbnails Area */}
            {capturedImages.length > 0 && (
                <div className="flex gap-2 overflow-x-auto p-2 scrollbar-hide">
                    {capturedImages.map((img, idx) => (
                        <div 
                            key={idx}
                            className={`relative shrink-0 w-12 h-12 rounded-md border-2 transition-all cursor-pointer ${idx === currentIndex ? 'border-accent-cyan' : 'border-transparent opacity-60 hover:opacity-100'}`}
                            onClick={() => onIndexChange(idx)}
                        >
                            <img src={img} className="w-full h-full object-cover rounded-[3px]" />
                            <button 
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onRemoveImage(idx);
                                }}
                                className="absolute -top-1.5 -right-1.5 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center text-white hover:bg-red-600 transition-colors"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                    ))}
                </div>
            )}

            <style>{`
                @keyframes scanMove {
                    0%, 100% { top: 10%; }
                    50% { top: 90%; }
                }
                .scrollbar-hide::-webkit-scrollbar {
                    display: none;
                }
                .scrollbar-hide {
                    -ms-overflow-style: none;
                    scrollbar-width: none;
                }
            `}</style>
        </div>
    );
}
