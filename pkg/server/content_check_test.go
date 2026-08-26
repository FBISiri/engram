package server

import "testing"

func defaultChecker() *ContentChecker {
	return &ContentChecker{MinContentLength: 20, MaxContentLength: 2000, RequireTags: true, RequireSourceType: true}
}

func hasType(advs []Advisory, typ string) bool {
	for _, a := range advs {
		if a.Type == typ {
			return true
		}
	}
	return false
}

func TestContentCheck_Short(t *testing.T) {
	advs := defaultChecker().Check("too short", []string{"t"}, "user_input")
	if !hasType(advs, "content_too_short") {
		t.Errorf("expected content_too_short, got %+v", advs)
	}
}

func TestContentCheck_Long(t *testing.T) {
	long := make([]byte, 2001)
	for i := range long {
		long[i] = 'a'
	}
	advs := defaultChecker().Check(string(long), []string{"t"}, "user_input")
	if !hasType(advs, "content_too_long") {
		t.Errorf("expected content_too_long, got %+v", advs)
	}
}

func TestContentCheck_MissingTags(t *testing.T) {
	advs := defaultChecker().Check("this content is definitely long enough now", nil, "user_input")
	if !hasType(advs, "tags_missing") {
		t.Errorf("expected tags_missing, got %+v", advs)
	}
}

func TestContentCheck_MissingSourceType(t *testing.T) {
	advs := defaultChecker().Check("this content is definitely long enough now", []string{"t"}, "")
	if !hasType(advs, "source_type_missing") {
		t.Errorf("expected source_type_missing, got %+v", advs)
	}
}

func TestContentCheck_AllOK(t *testing.T) {
	advs := defaultChecker().Check("this content is definitely long enough now", []string{"t"}, "user_input")
	if len(advs) != 0 {
		t.Errorf("expected no advisories, got %+v", advs)
	}
}

// TestContentCheck_Boundary: exactly MinContentLength and MaxContentLength are OK.
func TestContentCheck_Boundary(t *testing.T) {
	c := defaultChecker()
	exactMin := make([]byte, 20)
	for i := range exactMin {
		exactMin[i] = 'a'
	}
	if advs := c.Check(string(exactMin), []string{"t"}, "user_input"); hasType(advs, "content_too_short") {
		t.Errorf("len==min must not trigger short, got %+v", advs)
	}
	exactMax := make([]byte, 2000)
	for i := range exactMax {
		exactMax[i] = 'a'
	}
	if advs := c.Check(string(exactMax), []string{"t"}, "user_input"); hasType(advs, "content_too_long") {
		t.Errorf("len==max must not trigger long, got %+v", advs)
	}
}

// TestContentCheck_RuneBoundary: length is counted in runes, not bytes. A CJK
// string of 19 runes (57 bytes) triggers too_short; 20 runes does not; 2001
// runes triggers too_long and reports the rune count.
func TestContentCheck_RuneBoundary(t *testing.T) {
	c := defaultChecker()
	repeat := func(r rune, n int) string {
		rs := make([]rune, n)
		for i := range rs {
			rs[i] = r
		}
		return string(rs)
	}

	// 19 CJK runes → too short (byte len 57 would wrongly pass a byte check).
	if advs := c.Check(repeat('记', 19), []string{"t"}, "user_input"); !hasType(advs, "content_too_short") {
		t.Errorf("19 runes must trigger content_too_short, got %+v", advs)
	}
	// 20 CJK runes → OK.
	if advs := c.Check(repeat('记', 20), []string{"t"}, "user_input"); hasType(advs, "content_too_short") {
		t.Errorf("20 runes must not trigger content_too_short, got %+v", advs)
	}
	// 2001 CJK runes → too long, and content_length reports rune count.
	advs := c.Check(repeat('记', 2001), []string{"t"}, "user_input")
	if !hasType(advs, "content_too_long") {
		t.Fatalf("2001 runes must trigger content_too_long, got %+v", advs)
	}
	for _, a := range advs {
		if a.Type == "content_too_long" && a.Data["content_length"] != 2001 {
			t.Errorf("content_length should be rune count 2001, got %v", a.Data["content_length"])
		}
	}
}

// TestContentCheck_TogglesOff: disabled RequireTags/RequireSourceType suppress those advisories.
func TestContentCheck_TogglesOff(t *testing.T) {
	c := &ContentChecker{MinContentLength: 20, MaxContentLength: 2000, RequireTags: false, RequireSourceType: false}
	advs := c.Check("this content is definitely long enough now", nil, "")
	if len(advs) != 0 {
		t.Errorf("expected no advisories with toggles off, got %+v", advs)
	}
}
