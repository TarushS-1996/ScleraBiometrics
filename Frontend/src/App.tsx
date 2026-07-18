import { useState, useCallback } from 'react';
import Navbar from './components/Navbar';
import LiveCameraPreview from './components/LiveCameraPreview';
import CaptureLoadImage from './components/CaptureLoadImage';
import AnalysisResult from './components/AnalysisResult';
import ControlInterface from './components/ControlInterface';
import RecordScreen from './components/RecordScreen/index';
import { identifyUser, registerUser, runUnifiedLiveness } from './services/api';
import type { IdentifyResponse, SegmentResponse } from './types';
import NoMatchModal from './components/NoMatchModal';
import LivenessModal from './components/LivenessModal';

interface Toast {
  message: string;
  type: 'success' | 'error';
  id: number;
}

interface LivenessUiState {
  status: 'good' | 'incorrect';
  title: string;
  message: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'main' | 'record'>('main');
  const [capturedImages, setCapturedImages] = useState<string[]>([]);
  const [capturedFiles, setCapturedFiles] = useState<File[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [userId, setUserId] = useState('');
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [eyeSide, setEyeSide] = useState('right');
  const [analysisResult, setAnalysisResult] = useState<IdentifyResponse | SegmentResponse | null>(null);
  const [segmentResults, setSegmentResults] = useState<SegmentResponse[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [isRegistering, setIsRegistering] = useState(false);
  const [showNoMatchModal, setShowNoMatchModal] = useState(false);
  const [showLivenessModal, setShowLivenessModal] = useState(false);
  const [livenessUiState, setLivenessUiState] = useState<LivenessUiState>({
    status: 'incorrect',
    title: 'Liveness Not Verified',
    message: "Only one image; can't determine liveness.",
  });
  const [mode, setMode] = useState<'verify' | 'add'>('add');
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: 'success' | 'error') => {
    const id = Date.now();
    setToasts(prev => [...prev, { message, type, id }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3000);
  }, []);

  const handleNameChange = (fn: string, ln: string) => {
    setFirstName(fn);
    setLastName(ln);
    const cleanFn = fn.trim() || 'User';
    const cleanLn = ln.trim() || 'Name';
    setUserId(`${cleanFn}_${cleanLn}`);
  };

  const evaluateLiveness = useCallback(async (files: File[]) => {
    const response = await runUnifiedLiveness(files);
    console.log('[biometrics-debug] unified liveness response', response);

    if (response.input_count < 2) {
      return {
        pass: false,
        ui: {
          status: 'incorrect' as const,
          title: 'Liveness Not Verified',
          message: response.message || "Only one image; can't determine liveness.",
        },
      };
    }

    const finalStatus = response.pupil?.dilation?.first_to_last?.status;
    const ratio = response.pupil?.dilation?.first_to_last?.diameter_change_ratio;

    if (finalStatus === 'dilated' || finalStatus === 'constricted') {
      return {
        pass: true,
        ui: {
          status: 'good' as const,
          title: 'Liveness Verified',
          message: `Liveness status: ${finalStatus}${typeof ratio === 'number' ? ` (${(ratio * 100).toFixed(1)}%)` : ''}`,
        },
      };
    }

    const fallbackMessage =
      response.message ||
      (finalStatus === 'stable'
        ? 'Liveness status: stable pupil size; verification failed.'
        : 'Liveness status unavailable; verification failed.');

    return {
      pass: false,
      ui: {
        status: 'incorrect' as const,
        title: 'Liveness Not Verified',
        message: fallbackMessage,
      },
    };
  }, []);

  const performScan = useCallback(async (file: File, livenessFiles?: File[]) => {
    setAnalysisResult(null); // Clear previous result
    setSegmentResults([]); // Clear previous segment results
    if (mode === 'add') return;
    setIsScanning(true);

    console.log('[biometrics-debug] performScan started', {
      mode,
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
    });

    try {
      const response = await identifyUser(file);
      console.log('[biometrics-debug] performScan response', response);
      setAnalysisResult(response);

      console.log('[biometrics-debug] checking unified liveness after scan');
      let liveness;
      try {
        liveness = await evaluateLiveness(livenessFiles && livenessFiles.length ? livenessFiles : [file]);
      } catch (livenessErr) {
        console.error('[biometrics-debug] performScan liveness call failed', livenessErr);
        setLivenessUiState({
          status: 'incorrect',
          title: 'Liveness Check Error',
          message: 'Could not reach liveness API. Please try again.',
        });
        setShowLivenessModal(true);
        return;
      }

      console.log('[biometrics-debug] performScan liveness evaluated', liveness);
      setLivenessUiState(liveness.ui);
      addToast(`Liveness: ${liveness.ui.title}`, liveness.pass ? 'success' : 'error');
      if (!liveness.pass) {
        setShowLivenessModal(true);
        return;
      }

      if (response.best_match?.label === "DIFFERENT") {
        setShowNoMatchModal(true);
      } else {
        addToast('Identification complete', 'success');
      }
    } catch (err: unknown) {
      const message = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Identification failed';
      addToast(message, 'error');
      console.error('[biometrics-debug] performScan failed', err);
    } finally {
      setIsScanning(false);
    }
  }, [mode, addToast, evaluateLiveness]);

  const handleCapture = useCallback((dataUrl: string, file: File) => {
    console.log('[biometrics-debug] handleCapture', {
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type,
      mode,
    });

    setCapturedImages(prev => {
      const next = [...prev, dataUrl];
      setCurrentImageIndex(next.length - 1);
      return next;
    });
    setCapturedFiles(prev => [...prev, file]);
    addToast('Image captured successfully', 'success');
    if (mode === 'verify') {
      performScan(file);
    }
  }, [mode, performScan, addToast]);

  const handleBurstCapture = useCallback((captures: { dataUrl: string; file: File }[]) => {
    if (!captures.length) return;

    const dataUrls = captures.map(c => c.dataUrl);
    const files = captures.map(c => c.file);

    setCapturedImages(prev => {
      const newIndex = prev.length;
      const next = [...prev, ...dataUrls];
      setCurrentImageIndex(newIndex);
      return next;
    });

    setCapturedFiles(prev => [...prev, ...files]);
    addToast(`Burst captured: ${captures.length} images`, 'success');

    if (mode === 'verify') {
      performScan(files[0], files);
    }
  }, [mode, performScan, addToast]);

  const handleImageUpload = (dataUrls: string[], files: File[]) => {
    console.log('[biometrics-debug] handleImageUpload', {
      count: files.length,
      mode,
      files: files.map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
      })),
    });

    setCapturedImages(prev => {
      const newIndex = prev.length;
      const next = [...prev, ...dataUrls];
      setCurrentImageIndex(newIndex);
      return next;
    });
    setCapturedFiles(prev => [...prev, ...files]);
    addToast(`${files.length} image(s) uploaded successfully`, 'success');
    if (mode === 'verify' && files.length > 0) {
      performScan(files[0]);
    }
  };

  const handleStartScan = () => {
    if (capturedFiles.length > 0) {
      console.log('[biometrics-debug] handleStartScan', {
        selectedIndex: currentImageIndex,
        fileName: capturedFiles[currentImageIndex]?.name,
        fileSize: capturedFiles[currentImageIndex]?.size,
        fileType: capturedFiles[currentImageIndex]?.type,
      });
      performScan(capturedFiles[currentImageIndex]);
    }
  };

  const handleRegister = async (fn: string, ln: string) => {
    if (capturedFiles.length === 0 || !fn.trim() || !ln.trim()) {
      addToast('First Name, Last Name and at least one image are required', 'error');
      return;
    }
    setIsRegistering(true);
    setSegmentResults([]); // Clear previous results

    console.log('[biometrics-debug] handleRegister started', {
      firstName: fn,
      lastName: ln,
      userId: `${fn.trim() || 'User'}_${ln.trim() || 'Name'}`,
      eyeSide,
      imageCount: capturedFiles.length,
      files: capturedFiles.map(file => ({
        name: file.name,
        size: file.size,
        type: file.type,
      })),
    });

    try {
      console.log('[biometrics-debug] checking unified liveness before register');
      let liveness;
      try {
        liveness = await evaluateLiveness(capturedFiles);
      } catch (livenessErr) {
        console.error('[biometrics-debug] handleRegister liveness call failed', livenessErr);
        setLivenessUiState({
          status: 'incorrect',
          title: 'Liveness Check Error',
          message: 'Could not reach liveness API. Please try again.',
        });
        setShowLivenessModal(true);
        setIsRegistering(false);
        return;
      }

      console.log('[biometrics-debug] handleRegister liveness evaluated', liveness);
      setLivenessUiState(liveness.ui);
      addToast(`Liveness: ${liveness.ui.title}`, liveness.pass ? 'success' : 'error');
      if (!liveness.pass) {
        setShowLivenessModal(true);
        setIsRegistering(false);
        return;
      }

      const results: SegmentResponse[] = [];

      // Make separate API calls for each image
      for (let i = 0; i < capturedFiles.length; i++) {
        const file = capturedFiles[i];
        console.log('[biometrics-debug] registering image', {
          index: i,
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
        });

        const response = await registerUser(file, userId, eyeSide, fn, ln);

        results.push(response);
      }

      if (results.length > 0) {
        setSegmentResults(results);
        addToast(`${capturedFiles.length} image(s) registered successfully`, 'success');
      }
    } catch (err: unknown) {
      const message = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || 'Registration failed';
      addToast(message, 'error');
      console.error('[biometrics-debug] handleRegister failed', err);
    } finally {
      setIsRegistering(false);
    }
  };

  const removeImage = (index: number) => {
    setCapturedImages(prev => prev.filter((_, i) => i !== index));
    setCapturedFiles(prev => prev.filter((_, i) => i !== index));
    if (currentImageIndex >= capturedImages.length - 1 && currentImageIndex > 0) {
      setCurrentImageIndex(currentImageIndex - 1);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-bg-primary overflow-hidden font-inter">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

      {activeTab === 'main' ? (
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-[1.6fr_1fr] grid-rows-auto gap-4 p-4 overflow-y-auto ">
          <LiveCameraPreview onCapture={handleCapture} onBurstCapture={handleBurstCapture} />
          <CaptureLoadImage
            capturedImages={capturedImages}
            isScanning={isScanning}
            currentIndex={currentImageIndex}
            onIndexChange={setCurrentImageIndex}
            onRemoveImage={removeImage}
          />
          <AnalysisResult
            result={analysisResult}
            allResults={segmentResults}
            mode={mode}
            originalImages={capturedImages}
          />
          <ControlInterface
            userId={userId}
            firstName={firstName}
            lastName={lastName}
            eyeSide={eyeSide}
            onUserIdChange={setUserId}
            onNameChange={handleNameChange}
            onEyeSideChange={setEyeSide}
            onImageUpload={handleImageUpload}
            onStartScan={handleStartScan}
            onRegister={handleRegister}
            hasImage={capturedImages.length > 0}
            isScanning={isScanning}
            isRegistering={isRegistering}
            mode={mode}
            onModeChange={(m) => {
              setMode(m);
              setAnalysisResult(null);
              setSegmentResults([]);
            }}
          />
        </main>
      ) : (

        <RecordScreen />
      )}

      {/* Toast Notifications */}
      <div className="fixed top-5 left-1/2 -translate-x-1/2 z-2000 flex flex-col gap-2">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`px-6 py-2.5 rounded-lg text-[13px] font-medium animate-[fadeInOut_3s_ease_forwards] border ${toast.type === 'success'
              ? 'bg-accent-green/20 border-accent-green text-accent-green'
              : 'bg-accent-magenta/20 border-accent-magenta text-accent-magenta'
              }`}
          >
            {toast.message}
          </div>
        ))}
      </div>

      <NoMatchModal
        isOpen={showNoMatchModal}
        onClose={() => setShowNoMatchModal(false)}
        onTryAgain={() => {
          setShowNoMatchModal(false);
          handleStartScan();
        }}
        onAddNewUser={() => {
          setShowNoMatchModal(false);
          setMode('add');
        }}
      />
      <LivenessModal
        isOpen={showLivenessModal}
        onClose={() => setShowLivenessModal(false)}
        status={livenessUiState.status}
        title={livenessUiState.title}
        message={livenessUiState.message}
        onTryAgain={() => {
          setShowLivenessModal(false);
          if (mode === 'verify') {
            handleStartScan();
          } else {
            handleRegister(firstName, lastName);
          }
        }}
        onAddNewUser={() => {
          setShowLivenessModal(false);
          setMode('add');
        }}
      />

      <style>{`
        @keyframes fadeInOut {
          0% { opacity: 0; transform: translateY(10px); }
          15% { opacity: 1; transform: translateY(0); }
          85% { opacity: 1; }
          100% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}

export default App;
