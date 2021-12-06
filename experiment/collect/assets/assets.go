package assets

import (
	"embed"
	"log"
)

//go:embed *
var fs embed.FS

func Read(filename string) string {
	content, err := fs.ReadFile(filename)
	if err != nil {
		log.Fatalf("failed to read asset file: %s", filename)
	}
	return string(content)
}
