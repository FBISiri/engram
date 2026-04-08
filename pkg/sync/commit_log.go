package sync

import (
	"context"
	"encoding/json"
	"fmt"
	"math"

	bolt "go.etcd.io/bbolt"
)

var bucketName = []byte("commit_log")

// boltCommitLog implements CommitLog backed by BoltDB.
type boltCommitLog struct {
	db *bolt.DB
}

// NewBoltCommitLog opens (or creates) a BoltDB file and returns a CommitLog.
func NewBoltCommitLog(path string) (CommitLog, error) {
	db, err := bolt.Open(path, 0600, nil)
	if err != nil {
		return nil, fmt.Errorf("open bolt: %w", err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(bucketName)
		return err
	}); err != nil {
		db.Close()
		return nil, fmt.Errorf("create bucket: %w", err)
	}
	return &boltCommitLog{db: db}, nil
}

// tsKey encodes a float64 timestamp into an 8-byte big-endian key for ordered iteration.
func tsKey(ts float64) []byte {
	bits := math.Float64bits(ts)
	b := make([]byte, 8)
	for i := 7; i >= 0; i-- {
		b[i] = byte(bits & 0xff)
		bits >>= 8
	}
	return b
}

func tsFromKey(b []byte) float64 {
	var bits uint64
	for _, v := range b {
		bits = (bits << 8) | uint64(v)
	}
	return math.Float64frombits(bits)
}

// uniqueKey appends the entry ID to the timestamp key to avoid collisions.
func uniqueKey(entry CommitEntry) []byte {
	k := tsKey(entry.Timestamp)
	return append(k, []byte(entry.ID)...)
}

func (l *boltCommitLog) Append(_ context.Context, entry CommitEntry) error {
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	return l.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketName).Put(uniqueKey(entry), data)
	})
}

func (l *boltCommitLog) ReadSince(_ context.Context, since float64) ([]CommitEntry, error) {
	var entries []CommitEntry
	err := l.db.View(func(tx *bolt.Tx) error {
		c := tx.Bucket(bucketName).Cursor()
		start := tsKey(since)
		for k, v := c.Seek(start); k != nil; k, v = c.Next() {
			var e CommitEntry
			if err := json.Unmarshal(v, &e); err != nil {
				return err
			}
			if e.Timestamp >= since {
				entries = append(entries, e)
			}
		}
		return nil
	})
	return entries, err
}

func (l *boltCommitLog) Truncate(_ context.Context, before float64) error {
	return l.db.Update(func(tx *bolt.Tx) error {
		b := tx.Bucket(bucketName)
		c := b.Cursor()
		for k, _ := c.First(); k != nil; k, _ = c.Next() {
			ts := tsFromKey(k[:8])
			if ts >= before {
				break
			}
			if err := b.Delete(k); err != nil {
				return err
			}
		}
		return nil
	})
}

func (l *boltCommitLog) Close() error {
	return l.db.Close()
}
