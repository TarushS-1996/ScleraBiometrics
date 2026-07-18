import React, { useState } from 'react';
import { Settings, X, Sun, Video, Edit, Check, Loader2 } from 'lucide-react';
import { useCameras } from '../hooks/useCameras';

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsDrawer({ isOpen, onClose }: SettingsDrawerProps) {
  const { cameras: detectedCameras, isLoading, selectedCameraId, setSelectedCameraId } = useCameras();
  const [lighting, setLighting] = useState(30);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [customLabels, setCustomLabels] = useState<Record<string, string>>({});

  const [prevCameras, setPrevCameras] = useState(detectedCameras);

  // Sync custom labels with detected cameras initially
  if (detectedCameras !== prevCameras) {
    setPrevCameras(detectedCameras);
    setCustomLabels((prev) => {
      const next = { ...prev };
      let changed = false;
      detectedCameras.forEach((cam: { deviceId: string; label: string }) => {
        if (!next[cam.deviceId]) {
          next[cam.deviceId] = cam.label;
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }

  return (
    <>
      <div
        className={`fixed inset-0 bg-black/60 backdrop-blur-sm z-2000 transition-opacity duration-300 ${isOpen ? 'opacity-100 visible' : 'opacity-0 invisible'}`}
        onClick={onClose}
      />

      <div
        className={`fixed top-0 right-0 h-full w-[340px] bg-bg-secondary border-l border-border-custom z-2001 transform transition-transform duration-300 ease-in-out flex flex-col shadow-2xl ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="flex items-center justify-between p-5 border-b border-border-custom shadow-sm">
          <div className="flex items-center gap-3 text-text-primary">
            <Settings className="w-5 h-5 text-accent-cyan" />
            <span className="text-[17px] font-medium tracking-wide">Settings</span>
          </div>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors hover:bg-white/5 p-1 rounded-md"
          >
            <X className="w-[18px] h-[18px]" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-10">

          {/* Lighting Control */}
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-2.5 text-text-secondary">
              <Sun className="w-[18px] h-[18px]" />
              <span className="text-[14px]">Lighting Control</span>
            </div>

            <div className="flex flex-col gap-3 pl-7">
              <span className="text-[13px] text-text-muted">Lighting</span>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={lighting}
                  onChange={(e) => setLighting(parseInt(e.target.value))}
                  className="flex-1 h-1 bg-border-custom rounded-lg appearance-none cursor-pointer accent-accent-cyan outline-none"
                />
                <div className="bg-bg-primary border border-accent-cyan/40 text-text-primary text-[13px] px-3 py-1 rounded-md min-w-[44px] text-center font-medium">
                  {lighting}
                </div>
              </div>
            </div>
          </div>

          {/* Camera Connections */}
          <div className="flex flex-col gap-5">
            <div className="flex items-center gap-2.5 text-text-secondary">
              <Video className="w-[18px] h-[18px] text-accent-cyan" />
              <span className="text-[14px]">Available Cameras</span>
            </div>

            <div className="flex flex-col gap-3.5 pl-7">
              {isLoading ? (
                <div className="flex items-center gap-3 text-text-muted py-4">
                  <Loader2 className="w-5 h-5 animate-spin text-accent-cyan" />
                  <span className="text-[13px]">Detecting cameras...</span>
                </div>
              ) : detectedCameras.length === 0 ? (
                <div className="text-[13px] text-text-muted py-4 italic">
                  No cameras detected
                </div>
              ) : (
                detectedCameras.map((cam: { deviceId: string; label: string }, idx: number) => (
                  <div
                    key={cam.deviceId}
                    className={`flex items-center gap-3 relative p-1 rounded-lg transition-all border ${selectedCameraId === cam.deviceId ? 'bg-accent-blue/10 border-accent-blue/20' : 'bg-transparent border-transparent'}`}
                  >
                    {/* Selection Indicator Dot/Icon */}
                    <div
                      onClick={() => setSelectedCameraId(cam.deviceId)}
                      className={`w-4 h-4 rounded-full border flex items-center justify-center cursor-pointer transition-colors ${selectedCameraId === cam.deviceId ? 'bg-accent-blue border-accent-blue' : 'border-border-custom hover:border-accent-cyan'}`}
                    >
                      {selectedCameraId === cam.deviceId && <div className="w-1.5 h-1.5 bg-white rounded-full" />}
                    </div>

                    <input
                      type="text"
                      value={customLabels[cam.deviceId] || cam.label}
                      readOnly={editingIndex !== idx}
                      onClick={() => setSelectedCameraId(cam.deviceId)}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                        setCustomLabels((prev: Record<string, string>) => ({
                          ...prev,
                          [cam.deviceId]: e.target.value
                        }));
                      }}
                      className={`flex-1 bg-bg-dark border cursor-pointer ${editingIndex === idx ? 'border-accent-cyan' : 'border-border-custom/50'} text-text-secondary text-[13px] rounded-md px-3.5 py-2.5 focus:outline-none transition-colors`}
                    />
                    {editingIndex === idx ? (
                      <div className="flex items-center gap-1.5">
                        <button
                          onClick={() => setEditingIndex(null)}
                          className="w-6 h-6 flex items-center justify-center bg-accent-blue text-white rounded shrink-0 hover:bg-accent-blue/90 transition-colors"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => setEditingIndex(null)}
                          className="w-6 h-6 flex items-center justify-center bg-red-500 text-white rounded shrink-0 hover:bg-red-500/90 transition-colors"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => setEditingIndex(idx)}
                        className="text-text-muted hover:text-text-primary transition-colors p-1"
                      >
                        <Edit className="w-[18px] h-[18px]" />
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}
