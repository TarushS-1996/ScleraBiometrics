import { useEffect, useRef, useState, useCallback } from 'react';
import { useCameras } from '../hooks/useCameras';

interface LiveCameraPreviewProps {
    onCapture: (dataUrl: string, file: File) => void;
    onBurstCapture: (captures: { dataUrl: string; file: File }[]) => void;
}

export default function LiveCameraPreview({ onCapture, onBurstCapture }: LiveCameraPreviewProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const { cameras, isLoading, selectedCameraId, setSelectedCameraId } = useCameras();
    const [cameraActive, setCameraActive] = useState(false);

    const startCamera = useCallback(async (deviceId?: string) => {
        await Promise.resolve(); // Force async to avoid synchronous setState in effect warning
        try {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }

            const constraints: MediaStreamConstraints = {
                video: deviceId ? { deviceId: { exact: deviceId } } : true
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            streamRef.current = stream;
            setCameraActive(true);
        } catch (err) {
            console.error('Error accessing camera:', err);
            setCameraActive(false);
        }
    }, []);

    useEffect(() => {
        const initCamera = () => {
            if (selectedCameraId) {
                startCamera(selectedCameraId);
            } else if (cameras.length > 0) {
                startCamera(cameras[0].deviceId);
            } else {
                startCamera();
            }
        };

        const timeoutId = setTimeout(initCamera, 0);

        return () => {
            clearTimeout(timeoutId);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track: MediaStreamTrack) => track.stop());
            }
        };
    }, [startCamera, selectedCameraId, cameras]);


    const handleCameraSelect = (id: string) => {
        setSelectedCameraId(id);
        startCamera(id);
    };

    const handleCapture = () => {
        if (videoRef.current && cameraActive) {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            if (ctx) {
                ctx.drawImage(videoRef.current, 0, 0);
                const dataUrl = canvas.toDataURL('image/jpeg');

                canvas.toBlob((blob) => {
                    if (blob) {
                        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
                        onCapture(dataUrl, file);
                    }
                }, 'image/jpeg');
            }
        }
    };

    const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

    const handleBurstCapture = async () => {
        if (!videoRef.current || !cameraActive) return;

        const captures: { dataUrl: string; file: File }[] = [];
        const total = 5;

        for (let i = 0; i < total; i++) {
            if (!videoRef.current) break;

            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) continue;

            ctx.drawImage(videoRef.current, 0, 0);
            const dataUrl = canvas.toDataURL('image/jpeg');

            const blob = await new Promise<Blob | null>((resolve) => {
                canvas.toBlob((b) => resolve(b), 'image/jpeg');
            });

            if (blob) {
                const file = new File([blob], `burst_capture_${i + 1}.jpg`, { type: 'image/jpeg' });
                captures.push({ dataUrl, file });
            }

            if (i < total - 1) {
                await sleep(120);
            }
        }

        if (captures.length) {
            onBurstCapture(captures);
        }
    };

    return (
        <div className="bg-bg-secondary border border-border-custom rounded-xl p-4 flex flex-col overflow-hidden min-h-[280px]">
            <div className="flex items-center gap-2 mb-3">
                <svg className="w-[18px] h-[18px] text-accent-cyan shrink-0" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                </svg>
                <h3 className="text-sm font-semibold">Live Camera Preview</h3>
            </div>

            <div className="flex-1 relative bg-bg-dark rounded-lg overflow-hidden flex items-center justify-center">
                {isLoading ? (
                    <div className="flex flex-col items-center justify-center gap-3 text-text-muted">
                        <div className="w-8 h-8 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin" />
                        <span className="text-sm">Initializing camera...</span>
                    </div>
                ) : cameraActive ? (
                    <>
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-full object-cover"
                        />
                        <div className="absolute top-3 left-3 z-2">
                            <select
                                className="px-2.5 py-1 border border-accent-cyan rounded bg-black/60 text-text-primary text-[11px] cursor-pointer outline-none [&>option]:bg-bg-secondary"
                                value={selectedCameraId}
                                onChange={(e) => handleCameraSelect(e.target.value)}
                            >
                                {cameras.map((camera: { deviceId: string; label: string }, idx: number) => (
                                    <option key={camera.deviceId} value={camera.deviceId}>
                                        {camera.label || `Camera ${idx + 1}`}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex flex-col gap-2 z-2">
                            <button
                                className="w-9 h-9 border border-border-custom rounded-lg bg-bg-secondary/90 text-text-primary flex items-center justify-center cursor-pointer transition-all duration-200 hover:border-accent-cyan hover:bg-accent-cyan/10"
                                onClick={handleCapture}
                                title="Capture Image"
                                disabled={!cameraActive}
                            >
                                <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 9c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3zm0 8c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-10L10 5H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2h-4l-2-2z" />
                                </svg>
                            </button>
                            <button
                                className="w-9 h-9 border border-border-custom rounded-lg bg-bg-secondary/90 text-text-primary flex items-center justify-center cursor-pointer transition-all duration-200 hover:border-accent-cyan hover:bg-accent-cyan/10 text-[10px] font-bold"
                                onClick={handleBurstCapture}
                                title="Burst Capture (5 images)"
                                disabled={!cameraActive}
                            >
                                5x
                            </button>
                            <button
                                className="w-9 h-9 border border-border-custom rounded-lg bg-bg-secondary/90 text-text-primary flex items-center justify-center cursor-pointer transition-all duration-200 hover:border-accent-cyan hover:bg-accent-cyan/10"
                                onClick={() => startCamera(selectedCameraId)}
                                title="Refresh Camera"
                            >
                                <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M17.65 6.35c-1.63-1.63-3.94-2.57-6.48-2.22-3.41.47-6.24 3.22-6.84 6.62C3.55 15.44 7.14 20 12 20c3.41 0 6.34-2.14 7.5-5.12.19-.49-.13-1.01-.66-1.01h-.03c-.35 0-.66.21-.78.54C17.18 17.06 14.8 19 12 19c-3.87 0-7-3.13-7-7 0-3.37 2.39-6.18 5.56-6.81 1.7-.34 3.3.16 4.54 1.15l-1.83 1.83c-.32.32-.09.87.35.87H19c.55 0 1-.45 1-1V4.23c0-.45-.54-.67-.85-.35l-1.5 1.47z" />
                                </svg>
                            </button>
                        </div>

                        <div className="absolute border-2 border-red-600 pointer-events-none z-1 top-[15%] left-[30%] w-[40%] h-[70%] hidden"></div>
                    </>
                ) : (
                    <div className="flex flex-col items-center justify-center gap-3 text-text-muted text-sm">
                        <svg className="w-12 h-12 opacity-50" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z" />
                        </svg>
                        <span>Camera is inactive or not available</span>
                    </div>
                )}
            </div>
        </div>
    );
}
