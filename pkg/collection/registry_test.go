package collection

import "testing"

func TestRegistry_RegisterAndList(t *testing.T) {
	r := &Registry{collections: map[string]CollectionInfo{}}
	if err := r.Register("foo", nil); err != nil {
		t.Fatalf("first register: %v", err)
	}
	if err := r.Register("foo", nil); err == nil {
		t.Fatalf("duplicate register should fail")
	}
	if err := r.Register("", nil); err == nil {
		t.Fatalf("empty name should fail")
	}
	list := r.List()
	if len(list) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(list))
	}
}

func TestRegistry_Resolve(t *testing.T) {
	r := &Registry{collections: map[string]CollectionInfo{}}
	cases := map[string]string{
		"user":       CollectionUser,
		"agent-self": CollectionAgentSelf,
		"reflection": CollectionReflection,
		"pigo":       CollectionPigo,
		"":           CollectionUser, // default
		"unknown":    CollectionUser, // safe fallback
	}
	for in, want := range cases {
		if got := r.Resolve(in); got != want {
			t.Errorf("Resolve(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestRegistry_Init(t *testing.T) {
	r := &Registry{collections: map[string]CollectionInfo{}}
	r.Init()
	if _, ok := r.Get(CollectionUser); !ok {
		t.Errorf("Init missing %s", CollectionUser)
	}
	if _, ok := r.Get(CollectionAgentSelf); !ok {
		t.Errorf("Init missing %s", CollectionAgentSelf)
	}
	if _, ok := r.Get(CollectionReflection); !ok {
		t.Errorf("Init missing %s", CollectionReflection)
	}
	if _, ok := r.Get(CollectionPigo); !ok {
		t.Errorf("Init missing %s", CollectionPigo)
	}
	// idempotent
	r.Init()
	if len(r.List()) != 4 {
		t.Errorf("Init not idempotent: got %d entries", len(r.List()))
	}
}
