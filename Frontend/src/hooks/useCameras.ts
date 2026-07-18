import { useState, useEffect, useCallback } from 'react';

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export function useCameras() {
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [selectedCameraId, _setSelectedCameraId] = useState<string>(() => {
    return localStorage.getItem('preferred-camera-id') || '';
  });
  const [error, setError] = useState<Error | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const setSelectedCameraId = useCallback((id: string) => {
    localStorage.setItem('preferred-camera-id', id);
    _setSelectedCameraId(id);
    // Dispatch a storage event so other instances of the hook update
    window.dispatchEvent(new Event('storage'));
  }, []);

  const updateDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices
        .filter(device => device.kind === "videoinput")
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || 'Unknown Camera'
        }));
      
      setCameras(videoInputs);
      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to enumerate devices'));
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const initCameras = async () => {
      try {
        // Request permission first to get labels
        await navigator.mediaDevices.getUserMedia({ video: true });
        await updateDevices();
      } catch (err) {
        console.error("Camera access error:", err);
        setError(err instanceof Error ? err : new Error('Camera access denied'));
        setIsLoading(false);
      }
    };

    initCameras();

    // Sync state if localStorage changes in other components
    const handleStorageChange = () => {
      const id = localStorage.getItem('preferred-camera-id');
      if (id) _setSelectedCameraId(id);
    };

    window.addEventListener('storage', handleStorageChange);
    navigator.mediaDevices.addEventListener('devicechange', updateDevices);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      navigator.mediaDevices.removeEventListener('devicechange', updateDevices);
    };
  }, [updateDevices]);

  return { cameras, selectedCameraId, setSelectedCameraId, error, isLoading, refresh: updateDevices };
}
