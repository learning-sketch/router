//! Shared HTTP client construction with a resilient TLS root strategy.
//!
//! reqwest 0.13 defaults its TLS backend to rustls together with the
//! `rustls-platform-verifier`, which loads the operating system CA trust store
//! *while the client is being built*. On minimal hosts and containers that do
//! not ship a system trust store (no `ca-certificates` package), building the
//! client fails with:
//!
//! ```text
//! Failed to create HTTP client: builder error: No CA certificates were found from the system
//! ```
//!
//! This prevents the router from starting even though it usually communicates
//! with workers over plain HTTP. To stay usable on such images we transparently
//! fall back to a bundled set of Mozilla root certificates, which keeps HTTPS
//! certificate verification working without depending on the system trust store.

use std::sync::OnceLock;

use reqwest::{Certificate, ClientBuilder};

/// Build a [`reqwest::Client`] from a configuration closure.
///
/// The closure receives a fresh [`ClientBuilder`] and should apply any router
/// specific settings (timeouts, pool configuration, ...). It may be invoked
/// more than once, so it must not depend on external mutable state.
///
/// The client is first built using the platform/system CA trust store. If that
/// fails (typically because the host has no system CA certificates installed),
/// the client is rebuilt using bundled webpki root certificates so the router
/// can still start and serve plain-HTTP workers.
pub fn build_client<F>(configure: F) -> Result<reqwest::Client, reqwest::Error>
where
    F: Fn(ClientBuilder) -> ClientBuilder,
{
    match configure(reqwest::Client::builder()).build() {
        Ok(client) => Ok(client),
        Err(err) => {
            tracing::warn!(
                error = %err,
                "Failed to build HTTP client using the system CA trust store; \
                 falling back to bundled root certificates. Install the \
                 `ca-certificates` package on the host to use the system trust store."
            );
            configure(reqwest::Client::builder().tls_certs_only(bundled_roots().iter().cloned()))
                .build()
        }
    }
}

/// Build a default [`reqwest::Client`] with the same root-certificate fallback
/// behaviour as [`build_client`] but without any extra configuration.
///
/// This is a drop-in replacement for `reqwest::Client::new()` that does not
/// panic on hosts lacking a system CA trust store.
pub fn default_client() -> reqwest::Client {
    build_client(|builder| builder).unwrap_or_else(|err| {
        // Mirror the behaviour of `reqwest::Client::new()`, which panics on
        // failure. Reaching this point means even the bundled roots could not
        // be used, which indicates a misconfigured build rather than a missing
        // system trust store.
        panic!("Failed to create HTTP client: {err}");
    })
}

/// Lazily-decoded bundled Mozilla root certificates.
fn bundled_roots() -> &'static [Certificate] {
    static ROOTS: OnceLock<Vec<Certificate>> = OnceLock::new();
    ROOTS.get_or_init(|| {
        webpki_root_certs::TLS_SERVER_ROOT_CERTS
            .iter()
            .filter_map(|cert| Certificate::from_der(cert.as_ref()).ok())
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_roots_are_available() {
        assert!(
            !bundled_roots().is_empty(),
            "expected bundled webpki root certificates to be decodable"
        );
    }

    #[test]
    fn build_client_succeeds() {
        let client = build_client(|builder| builder.timeout(std::time::Duration::from_secs(5)));
        assert!(client.is_ok());
    }

    #[test]
    fn default_client_does_not_panic() {
        let _ = default_client();
    }

    #[test]
    fn fallback_builder_uses_bundled_roots() {
        // Directly exercise the configuration used by the fallback path to
        // ensure a client can be built purely from bundled roots, without
        // touching the system trust store (the scenario that fails with
        // "No CA certificates were found from the system").
        let client = reqwest::Client::builder()
            .tls_certs_only(bundled_roots().iter().cloned())
            .build();
        assert!(
            client.is_ok(),
            "expected the bundled-roots fallback client to build: {:?}",
            client.err()
        );
    }
}
