class Hivebear < Formula
  desc "AI that fits your machine — run LLMs on any device regardless of GPU"
  homepage "https://github.com/BeckhamLabsLLC/HiveBear"
  version "0.1.7"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "08b8da81e655c0d8cdb56b12735a807a36511e93d4111601176515be560bbfa1"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "6fdd18f4a9526e56632f73169fda147187c69a892eebe348031fe12fd8b8176e"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "cdbf50e6f36035856ebaae5fcf8bacce10a375b3f79a3cc291822a8e4430723e"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "3fe06311bcde2fef53cf9481628823e2fb39dccdb26b62ebdd28ca084a228e3e"
    end
  end

  def install
    bin.install "hivebear"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/hivebear --version")
  end

  def caveats
    <<~EOS
      Get started with HiveBear:

        hivebear quickstart

      This will profile your hardware, recommend the best model,
      download it, and start an interactive chat session.
    EOS
  end
end
