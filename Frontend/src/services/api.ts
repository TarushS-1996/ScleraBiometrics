import axios from 'axios';
import type { IdentifyResponse, SegmentResponse, LogsResponse } from '../types';

const DEFAULT_API_BASE = 'http://127.0.0.1:8000';
export const API_BASE = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE).replace(/\/$/, '');

export async function identifyUser(imageFile: File): Promise<IdentifyResponse> {
    console.log('[biometrics-debug] identifyUser started', {
        url: `${API_BASE}/identify`,
        fileName: imageFile.name,
        fileSize: imageFile.size,
        fileType: imageFile.type,
    });

    const formData = new FormData();
    formData.append('image', imageFile);

    try {
        const response = await axios.post<IdentifyResponse>(`${API_BASE}/identify`, formData);
        console.log('[biometrics-debug] identifyUser success', {
            status: response.status,
            data: response.data,
        });
        clearLogsCache();
        return response.data;
    } catch (error: unknown) {
        console.error('[biometrics-debug] identifyUser failed', error);
        throw error;
    }
}

export async function registerUser(
    imageFile: File,
    userId: string,
    eyeSide: string,
    firstName: string,
    lastName: string
): Promise<SegmentResponse> {
    console.log('[biometrics-debug] registerUser started', {
        url: `${API_BASE}/segment`,
        fileName: imageFile.name,
        fileSize: imageFile.size,
        fileType: imageFile.type,
        userId,
        eyeSide,
        firstName,
        lastName,
    });

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('user_id', userId);
    formData.append('eye_side', eyeSide);
    formData.append('first_name', firstName);
    formData.append('last_name', lastName);

    try {
        const response = await axios.post<SegmentResponse>(`${API_BASE}/segment`, formData);
        console.log('[biometrics-debug] registerUser success', {
            status: response.status,
            data: response.data,
        });
        clearLogsCache();
        return response.data;
    } catch (error: unknown) {
        console.error('[biometrics-debug] registerUser failed', error);
        throw error;
    }
}

export interface LogFilters {
    action?: 'new_user' | 'match';
    user_id?: string;
    matched?: boolean;
    from_date?: string;
    to_date?: string;
    limit?: number;
    offset?: number;
}

const logsCache = new Map<string, LogsResponse>();

export function clearLogsCache() {
    logsCache.clear();
}

export async function getLogs(filters: LogFilters = {}): Promise<LogsResponse> {
    const cacheKey = JSON.stringify(filters);
    if (logsCache.has(cacheKey)) {
        return logsCache.get(cacheKey)!;
    }

    const response = await axios.get<LogsResponse>(`${API_BASE}/logs`, {
        params: filters
    });
    
    logsCache.set(cacheKey, response.data);
    return response.data;
}

export async function exportLogs(): Promise<void> {
    const response = await axios.get(`${API_BASE}/logs/export`, {
        responseType: 'blob',
    });

    // Create a link element and trigger download
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `logs_${new Date().toISOString()}.csv`); // Assuming CSV, or check headers
    document.body.appendChild(link);
    link.click();
    link.remove();
}

export async function deleteLog(recordId: string): Promise<void> {
    await axios.delete(`${API_BASE}/logs/${recordId}`);
    clearLogsCache();
}

export function getImageUrl(userId: string, eyeSide: string, sample: string): string {
    return `${API_BASE}/image/${userId}/${eyeSide}/${sample}`;
}

const imageCache = new Map<string, string>();

export async function fetchImage(userId: string, eyeSide: string, sample: string): Promise<string> {
    const cacheKey = `${userId}_${eyeSide}_${sample}`;
    if (imageCache.has(cacheKey)) {
        return imageCache.get(cacheKey)!;
    }

    const response = await axios.get<{ image: string; path: string }>(`${API_BASE}/image/${encodeURIComponent(userId)}/${encodeURIComponent(eyeSide)}/${encodeURIComponent(sample)}`);
    const base64Data = response.data.image;
    const dataUrl = `data:image/png;base64,${base64Data}`;
    
    imageCache.set(cacheKey, dataUrl);
    return dataUrl;
}

export async function checkLiveness(): Promise<{ ok: boolean }> {
    console.log('[biometrics-debug] checkLiveness started', { url: `${API_BASE}/liveness_check` });

    try {
        const response = await axios.get<{ ok: boolean }>(`${API_BASE}/liveness_check`);
        console.log('[biometrics-debug] checkLiveness success', response.data);
        return response.data;
    } catch (error: unknown) {
        console.error('[biometrics-debug] checkLiveness failed', error);
        throw error;
    }
}

export interface UnifiedLivenessResponse {
    endpoint: string;
    input_count: number;
    message: string;
    pipeline: string[];
    pupil: {
        per_image: Array<{
            index: number;
            filename: string;
            pupil_class_id: number | null;
            confidence: number;
            used_fallback: boolean;
            metrics: {
                area_px: number;
                diameter_px: number;
                center: number[] | null;
                bbox: number[] | null;
            };
            segmentation_mask: string;
            pupil_mask: string;
            pupil_overlay: string;
        }>;
        dilation: {
            available: boolean;
            threshold: number;
            first_to_last: {
                from_index: number;
                to_index: number;
                status: string;
                diameter_change_ratio?: number;
                reason?: string;
            } | null;
            pairwise: Array<{
                from_index: number;
                to_index: number;
                status: string;
                diameter_change_ratio?: number;
                reason?: string;
            }>;
        };
    };
    vein_flow: {
        implemented: boolean;
        message: string;
    };
}

export async function runUnifiedLiveness(
    imageFiles: File[],
    changeThresholdRatio = 0.08,
): Promise<UnifiedLivenessResponse> {
    if (!imageFiles.length) {
        throw new Error('At least one image is required for liveness');
    }

    const formData = new FormData();
    for (const file of imageFiles) {
        formData.append('images', file);
    }
    formData.append('change_threshold_ratio', String(changeThresholdRatio));

    console.log('[biometrics-debug] runUnifiedLiveness started', {
        url: `${API_BASE}/liveness`,
        count: imageFiles.length,
        files: imageFiles.map(file => ({
            name: file.name,
            size: file.size,
            type: file.type,
        })),
        changeThresholdRatio,
    });

    try {
        const response = await axios.post<UnifiedLivenessResponse>(`${API_BASE}/liveness`, formData);
        console.log('[biometrics-debug] runUnifiedLiveness success', response.data);
        return response.data;
    } catch (error: unknown) {
        console.error('[biometrics-debug] runUnifiedLiveness failed', error);
        throw error;
    }
}
