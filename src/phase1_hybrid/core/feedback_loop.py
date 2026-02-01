import sqlite3
import re
from typing import List, Tuple

class FeedbackLoop:
    """
    Handles the propagation of user feedback (Accept/Reject/Modify) 
    to other pending sentences in the dataset.
    Strategy: 'Teach One, Fix All'
    """
    def __init__(self, db_manager):
        self.db = db_manager

    def propagate_rejection(self, text: str, label: str) -> int:
        """
        Removes all occurrences of the rejected entity (text + label) 
        from PENDING sentences' proposed annotations.
        Also adds to Denylist for future predictions.
        """
        if not text: return 0
        try:
            # 1. Add to Persistent Denylist
            self.db.conn.execute("INSERT OR IGNORE INTO denylist (text, label) VALUES (?, ?)", (text, label))

            # 2. Delete annotations for Pending sentences matching the bad pattern
            cursor = self.db.conn.execute("""
                DELETE FROM annotations 
                WHERE text_span = ? 
                  AND label = ? 
                  AND is_accepted = 0 
                  AND sentence_id IN (SELECT id FROM sentences WHERE status = 'pending')
            """, (text, label))
            
            # [NEW] Also delete even if accepted=1? (Risky if user changed mind?)
            # User implies rigorous propagation. If I reject X here, X should be gone everywhere pending.
            # But let's stick to is_accepted=0 for safety unless explicitly requested.
            
            count = cursor.rowcount
            self.db.conn.commit()
            return count
        except Exception as e:
            print(f"❌ Rejection Propagation Error: {e}")
            return 0
            
    def propagate_exact_confirmation(self, text: str, label: str) -> int:
        """
        [NEW] Marks ALL pre-existing pending annotations of the same text+label as ACCEPTED.
        Also adds to Persistent Allowlist so future predictions are auto-accepted.
        """
        if not text: return 0
        try:
            # 1. Add to Persistent Allowlist (Future Proofing)
            self.db.conn.execute("INSERT OR IGNORE INTO allowlist (text, label) VALUES (?, ?)", (text, label))
            
            # 2. Update existing Annotations (Pre-calculated ones)
            cursor = self.db.conn.execute("""
                UPDATE annotations 
                SET is_accepted = 1, confidence = 1.0, trigger_text = 'Auto-Confirmed Exact'
                WHERE text_span = ? 
                  AND label = ?
                  AND sentence_id IN (SELECT id FROM sentences WHERE status = 'pending')
            """, (text, label))
            
            count = cursor.rowcount
            self.db.conn.commit()
            return count
        except Exception as e:
            print(f"❌ Confirmation Propagation Error: {e}")
            return 0

    def propagate_boundary_fix(self, bad_span: str, good_span: str, label: str) -> int:
        """
        Updates annotations in PENDING sentences where the 'bad_span' is found,
        replacing it with 'good_span'.
        Also adds to Fix Rules for future predictions.
        """
        if not bad_span or not good_span: return 0
        try:
            # 1. Add to Persistent Fix Rules
            self.db.conn.execute("INSERT OR REPLACE INTO fix_rules (bad_text, good_text, label) VALUES (?, ?, ?)", (bad_span, good_span, label))

            # 2. Simple Update: Just change the stored span text in annotations.
            # [Aggressive Mode] Also mark as ACCEPTED because the user explicitly defined this fix.
            cursor = self.db.conn.execute("""
                UPDATE annotations 
                SET text_span = ?, trigger_text = 'Auto-Fix Propagated', is_accepted = 1, confidence = 1.0
                WHERE text_span = ? 
                  AND label = ?
                  AND sentence_id IN (SELECT id FROM sentences WHERE status = 'pending')
            """, (good_span, bad_span, label))
            
            count = cursor.rowcount
            self.db.conn.commit()
            return count
        except Exception as e:
            print(f"❌ Boundary Propagation Error: {e}")
            return 0

    def propagate_discovery(self, text: str, label: str) -> int:
        """
        Scans ALL PENDING sentences for the 'text'.
        If found, creates a new annotation automatically.
        """
        if not text or len(text) < 3: return 0 # Safety check
        
        try:
            # 1. Get all pending sentences that contain the string (Fast SQL LIKE)
            cursor = self.db.conn.execute("""
                SELECT id, text FROM sentences 
                WHERE status = 'pending' 
                  AND text LIKE ?
            """, (f"%{text}%",))
            
            candidates = cursor.fetchall()
            count = 0
            
            for row in candidates:
                sid = row['id']
                sent_text = row['text']
                
                # Check if annotation already exists
                exists = self.db.conn.execute("""
                    SELECT 1 FROM annotations 
                    WHERE sentence_id = ? AND text_span = ? AND label = ?
                """, (sid, text, label)).fetchone()
                
                if exists:
                    # [NEW] If it exists but wasn't accepted, accept it now!
                    self.db.conn.execute("""
                        UPDATE annotations 
                        SET is_accepted = 1, confidence = 1.0, trigger_text = 'Auto-Confirmed Propagated'
                        WHERE sentence_id = ? AND text_span = ? AND label = ?
                    """, (sid, text, label))
                    continue
                
                # Double check boundaries (don't match "cat" inside "catch") if needed?
                # For now, simplistic string check is what the user wants for "easy" speed.
                
                # Insert
                self.db.conn.execute("""
                    INSERT INTO annotations 
                    (sentence_id, text_span, label, confidence, source_sentence, trigger_text, is_accepted)
                    VALUES (?, ?, ?, 0.99, ?, 'Auto-Discovery Propagated', 1)
                """, (sid, text, label, sent_text))
                count += 1
            
            self.db.conn.commit()
            return count
            
        except Exception as e:
            print(f"❌ Discovery Propagation Error: {e}")
            return 0
