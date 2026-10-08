class Hivebear < Formula
  desc "Run local AI models, with picks matched to your hardware"
  homepage "https://hivebear.com"
  version "0.1.9"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "e6777d871f37745dd4e7cc78f547d8c12c6276d9b9d9a38db1ea1e565af26ae3"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "eaf1aab55ca5481a8df100a31fa9f819ca907b457f9358560342694d2ea2c0e9"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "86c270a159f5ea6c576f5c357c07a5aa2caef7177de5b3a68f1906e58f1cfe2f"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "551b4161c11b67e0d17d99fd2c63168faaf1bc2a9e96217d582d9eed12bcc35f"
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
