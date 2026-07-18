export interface SegmentResponse {
    message: string;
    user_id: string;
    eye_side: string;
    sample: string;
    total_samples: number;
    processed_image: string;
}

export interface BestMatch {
    user_id: string;
    eye_side: string;
    sample: string;
    similarity: number;
    matched_image: string;
    label: string;
}

export interface IdentifyResponse {
    best_match: BestMatch;
    processed_query_image: string;
}

export interface NewUserLog {
    action: 'new_user';
    user_id: string;
    eye_side: string;
    sample: string;
    total_samples: number;
    save_path: string;
    record_id: string;
    timestamp: string;
}

export interface MatchLog {
    action: 'match';
    matched: boolean;
    best_match_user_id: string;
    best_match_eye_side: string;
    best_match_sample: string;
    best_match_similarity: number;
    total_matches: number;
    record_id: string;
    timestamp: string;
}

export type LogEntry = NewUserLog | MatchLog;

export interface LogsResponse {
    total: number;
    offset: number;
    limit: number;
    logs: LogEntry[];
}
