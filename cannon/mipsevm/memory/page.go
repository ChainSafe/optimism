package memory

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sync"
)

var zlibWriterPool = sync.Pool{
	New: func() any {
		var buf bytes.Buffer
		return zlib.NewWriter(&buf)
	},
}

type Page []byte

func (p Page) MarshalJSON() ([]byte, error) { // nosemgrep
	var out bytes.Buffer
	w := zlibWriterPool.Get().(*zlib.Writer)
	defer zlibWriterPool.Put(w)
	w.Reset(&out)
	if _, err := w.Write(p[:]); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return json.Marshal(out.Bytes())
}

func (p Page) UnmarshalJSON(dat []byte) error {
	// Strip off the `"` characters at the start & end.
	dat = dat[1 : len(dat)-1]
	// Decode b64 then decompress
	r, err := zlib.NewReader(base64.NewDecoder(base64.StdEncoding, bytes.NewReader(dat)))
	if err != nil {
		return err
	}
	defer r.Close()
	if n, err := r.Read(p[:]); n != PageSize {
		return fmt.Errorf("epxeted %d bytes, but got %d", PageSize, n)
	} else if err == io.EOF {
		return nil
	} else {
		return err
	}
}

func (p Page) UnmarshalText(dat []byte) error {
	if len(dat) != PageSize*2 {
		return fmt.Errorf("expected %d hex chars, but got %d", PageSize*2, len(dat))
	}
	_, err := hex.Decode(p[:], dat)
	return err
}

type CachedPage struct {
	Data Page
	// intermediate nodes only
	Cache [PageSize / 32][32]byte
	// bit set to 1 if the intermediate node is valid
	OkLow, OkHigh uint64 // size is PageSize / 32 == 64 + 64 == 128
}

func (p *CachedPage) invalidate(pageAddr Word) {
	if pageAddr >= PageSize {
		panic("invalid page addr")
	}

	k := uint64((1 << PageAddrSize) | pageAddr)
	k >>= 5 + 1

	for k > 0 {
		mask := uint64(1) << (k & 63)   // Compute bit mask
		isHigh := k >> 6                // 1 if k >= 64, 0 if k < 64
		p.OkLow &^= mask * (1 - isHigh) // Zero out bit in OkLow if isHigh == 0
		p.OkHigh &^= mask * isHigh      // Zero out bit in OkHigh if isHigh == 1

		k >>= 1
	}
}

func (p *CachedPage) getBit(k uint64) bool {
	if k < 64 {
		return (p.OkLow & (1 << k)) != 0
	}
	return (p.OkHigh & (1 << (k - 64))) != 0
}

func (p *CachedPage) setBit(k uint64) {
	mask := uint64(1) << (k & 63) // Create bit mask
	isHigh := k >> 6              // 1 if k >= 64, 0 if k < 64

	// Branchless update
	p.OkLow |= mask * (1 - isHigh) // Set bit in OkLow if isHigh == 0
	p.OkHigh |= mask * isHigh      // Set bit in OkHigh if isHigh == 1
}

func (p *CachedPage) InvalidateFull() {
	p.OkHigh = 0
	p.OkLow = 0
	// p.Ok = bits.NewBitlist64(PageSize / 32)
}

func (p *CachedPage) MerkleRoot() [32]byte {
	// hash the bottom layer
	for i := uint64(0); i < PageSize; i += 64 {
		j := PageSize/32/2 + i/64
		if p.getBit(j) {
			continue
		}
		HashData(&p.Cache[j], p.Data[i:i+64])
		// p.Cache[j] = crypto.Keccak256Hash(p.Data[i : i+64 : i+64])
		//fmt.Printf("0x%x 0x%x -> 0x%x\n", p.Data[i:i+32], p.Data[i+32:i+64], p.Cache[j])
		// p.Ok[j] = true
		p.setBit(j)
	}

	// hash the cache layers
	for i := PageSize/32 - 2; i > 0; i -= 2 {
		j := i >> 1
		if p.getBit(uint64(j)) {
			continue
		}
		Hash2(&p.Cache[j], &p.Cache[i], &p.Cache[i+1])
		// p.Ok[j] = true
		p.setBit(uint64(j))
	}

	return p.Cache[1]
}

func (p *CachedPage) MerkleizeSubtree(gindex uint64) [32]byte {
	_ = p.MerkleRoot() // fill cache
	if gindex >= PageSize/32 {
		if gindex >= PageSize/32*2 {
			panic("gindex too deep")
		}
		// it's pointing to a bottom node
		nodeIndex := gindex & (PageAddrMask >> 5)
		return *(*[32]byte)(p.Data[nodeIndex*32 : nodeIndex*32+32 : nodeIndex*32+32])
	}
	return p.Cache[gindex]
}

func (p *CachedPage) MerkleizeNode(addr Word, gindex uint64) [32]byte {
	_ = p.MerkleRoot() // fill cache
	if gindex >= PageSize/32 {
		if gindex >= PageSize/32*2 {
			panic("gindex too deep")
		}

		// it's pointing to a bottom node
		nodeIndex := gindex & (PageAddrMask >> 5)
		return *(*[32]byte)(p.Data[nodeIndex*32 : nodeIndex*32+32])
	}
	return p.Cache[gindex]
}

func (p *CachedPage) GenerateProof(addr Word) [][32]byte {
	// Page-level proof
	pageGindex := PageSize>>5 + (addr&PageAddrMask)>>5

	proofs := make([][32]byte, 8)
	proofIndex := 0

	proofs[proofIndex] = p.MerkleizeSubtree(uint64(pageGindex))

	for idx := pageGindex; idx > 1; idx >>= 1 {
		sibling := idx ^ 1
		proofIndex++
		proofs[proofIndex] = p.MerkleizeSubtree(uint64(sibling))
	}

	return proofs
}
