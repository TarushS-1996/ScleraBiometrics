export interface RecordItem {
  id: string;
  userId: string;
  userName: string;
  actionType: string;
  result: 'Matched' | 'Unmatched';
  liveliness: string;
  confidence: string;
  timestamp: string;
}
