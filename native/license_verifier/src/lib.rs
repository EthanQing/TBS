use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use pyo3::create_exception;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::Value;

const PUBLIC_KEY_B64: &str = "qk/C58JiQDFp8UfxCp1TX+ABNZkD4yq+NsZ2LjNHuHE=";
const DEFAULT_LICENSE_PATH: &str = "/app/license/license.dat";
const LICENSE_DATA_ENV: &str = "TRAIN_PLATFORM_LICENSE_DATA";
const LICENSE_DATA_B64_ENV: &str = "TRAIN_PLATFORM_LICENSE_DATA_B64";

create_exception!(license, LicenseError, PyRuntimeError);

#[pyclass(frozen, module = "train_platform.core.license")]
#[derive(Clone, Debug)]
struct LicenseInfo {
    #[pyo3(get)]
    customer: String,
    #[pyo3(get)]
    deployment: String,
    expires_at_rfc3339: String,
}

#[pymethods]
impl LicenseInfo {
    #[getter]
    fn expires_at(&self, py: Python<'_>) -> PyResult<PyObject> {
        let datetime = py.import("datetime")?.getattr("datetime")?;
        datetime
            .call_method1("fromisoformat", (&self.expires_at_rfc3339,))
            .map(|value| value.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "LicenseInfo(customer={:?}, deployment={:?}, expires_at={})",
            self.customer, self.deployment, self.expires_at_rfc3339
        )
    }
}

#[derive(Clone, Debug)]
struct CachedLicense {
    source: Vec<u8>,
    info: LicenseInfo,
    expires_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
struct LicenseDocument {
    payload: Value,
    signature: String,
}

static CACHE: OnceLock<Mutex<Option<CachedLicense>>> = OnceLock::new();

fn cache() -> &'static Mutex<Option<CachedLicense>> {
    CACHE.get_or_init(|| Mutex::new(None))
}

fn env_enabled(name: &str, default: bool) -> bool {
    let default_value = if default { "1" } else { "0" };
    let value = env::var(name).unwrap_or_else(|_| default_value.to_string());
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y" | "on"
    )
}

#[pyfunction]
fn license_required() -> bool {
    cfg!(feature = "enforce-license") || env_enabled("TRAIN_PLATFORM_LICENSE_REQUIRED", false)
}

#[pyfunction]
fn assert_valid_license() -> PyResult<Option<LicenseInfo>> {
    if !license_required() {
        return Ok(None);
    }

    let source = load_license_bytes().map_err(license_error)?;
    let now = Utc::now();
    let mut guard = cache()
        .lock()
        .map_err(|_| license_error("License cache lock is poisoned.".to_string()))?;

    if let Some(cached) = guard.as_ref() {
        if cached.source == source {
            ensure_not_expired(cached.expires_at, now).map_err(license_error)?;
            return Ok(Some(cached.info.clone()));
        }
    }

    let public_key = production_public_key().map_err(license_error)?;
    let verified = verify_license(&source, now, &public_key).map_err(license_error)?;
    let info = verified.info.clone();
    *guard = Some(verified);
    Ok(Some(info))
}

fn load_license_bytes() -> Result<Vec<u8>, String> {
    if let Ok(value) = env::var(LICENSE_DATA_B64_ENV) {
        let value = value.trim();
        if !value.is_empty() {
            return decode_license_data_b64(value);
        }
    }

    if let Ok(value) = env::var(LICENSE_DATA_ENV) {
        if !value.trim().is_empty() {
            return Ok(value.into_bytes());
        }
    }

    let path = env::var("TRAIN_PLATFORM_LICENSE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LICENSE_PATH));
    read_license_file(&path)
}

fn decode_license_data_b64(value: &str) -> Result<Vec<u8>, String> {
    STANDARD
        .decode(value.as_bytes())
        .map_err(|_| format!("Invalid base64 value for {LICENSE_DATA_B64_ENV}."))
}

fn read_license_file(path: &PathBuf) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|_| {
        format!(
            "License file not found: {}. Set {LICENSE_DATA_B64_ENV}, TRAIN_PLATFORM_LICENSE_PATH, or mount license.dat into /app/license.",
            path.display()
        )
    })
}

fn production_public_key() -> Result<VerifyingKey, String> {
    let bytes = STANDARD
        .decode(PUBLIC_KEY_B64.as_bytes())
        .map_err(|_| "Invalid base64 value for public key.".to_string())?;
    let key: [u8; 32] = bytes
        .try_into()
        .map_err(|_| "Invalid public key length.".to_string())?;
    VerifyingKey::from_bytes(&key).map_err(|_| "Invalid public key.".to_string())
}

fn verify_license(
    source: &[u8],
    now: DateTime<Utc>,
    public_key: &VerifyingKey,
) -> Result<CachedLicense, String> {
    let document: LicenseDocument = serde_json::from_slice(source)
        .map_err(|error| format!("Invalid license file format: {error}"))?;
    if !document.payload.is_object() || document.signature.trim().is_empty() {
        return Err("Invalid license file: expected payload and signature.".to_string());
    }

    let payload_bytes = canonical_payload_bytes(&document.payload)?;
    let signature_bytes = STANDARD
        .decode(document.signature.as_bytes())
        .map_err(|_| "Invalid base64 value for signature.".to_string())?;
    let signature = Signature::from_slice(&signature_bytes)
        .map_err(|_| "Invalid license signature.".to_string())?;
    public_key
        .verify(&payload_bytes, &signature)
        .map_err(|_| "Invalid license signature.".to_string())?;

    let payload = document
        .payload
        .as_object()
        .ok_or_else(|| "Invalid license payload.".to_string())?;
    let customer = required_string(payload, "customer")?;
    let deployment = required_string(payload, "deployment")?;
    let expires_at_raw = required_string(payload, "expires_at")?;
    let expires_at = parse_datetime(&expires_at_raw)?;
    ensure_not_expired(expires_at, now)?;

    Ok(CachedLicense {
        source: source.to_vec(),
        info: LicenseInfo {
            customer,
            deployment,
            expires_at_rfc3339: expires_at.to_rfc3339(),
        },
        expires_at,
    })
}

fn required_string(
    payload: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<String, String> {
    let value = payload
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if value.is_empty() {
        return Err(format!("Invalid license payload: {field} is required."));
    }
    Ok(value)
}

fn canonical_payload_bytes(payload: &Value) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    write_canonical_json(payload, &mut output)?;
    Ok(output)
}

fn parse_datetime(value: &str) -> Result<DateTime<Utc>, String> {
    let raw = value.trim();
    if raw.is_empty() {
        return Err("Invalid license payload: expires_at is required.".to_string());
    }
    if let Ok(parsed) = DateTime::parse_from_rfc3339(raw) {
        return Ok(parsed.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(parsed) = NaiveDateTime::parse_from_str(raw, format) {
            return Ok(parsed.and_utc());
        }
    }
    if let Ok(parsed) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        return Ok(parsed
            .and_hms_opt(0, 0, 0)
            .expect("midnight is valid")
            .and_utc());
    }
    Err(format!("Invalid datetime for expires_at: {value}"))
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), String> {
    match value {
        Value::Object(map) => {
            output.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|error| format!("Invalid license payload: {error}"))?;
                output.push(b':');
                write_canonical_json(&map[key], output)?;
            }
            output.push(b'}');
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(item, output)?;
            }
            output.push(b']');
        }
        _ => serde_json::to_writer(output, value)
            .map_err(|error| format!("Invalid license payload: {error}"))?,
    }
    Ok(())
}

fn ensure_not_expired(expires_at: DateTime<Utc>, now: DateTime<Utc>) -> Result<(), String> {
    if expires_at <= now {
        return Err(format!("License expired at {}.", expires_at.to_rfc3339()));
    }
    Ok(())
}

fn license_error(message: String) -> PyErr {
    LicenseError::new_err(message)
}

#[pymodule]
fn license(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("LicenseError", module.py().get_type::<LicenseError>())?;
    module.add_class::<LicenseInfo>()?;
    module.add_function(wrap_pyfunction!(license_required, module)?)?;
    module.add_function(wrap_pyfunction!(assert_valid_license, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use ed25519_dalek::{Signer, SigningKey};
    use serde_json::json;

    fn signed_license(payload: Value) -> (Vec<u8>, VerifyingKey) {
        let signing_key = SigningKey::from_bytes(&[7_u8; 32]);
        let payload_bytes = canonical_payload_bytes(&payload).unwrap();
        let signature = signing_key.sign(&payload_bytes);
        let document = json!({
            "payload": payload,
            "signature": STANDARD.encode(signature.to_bytes()),
        });
        (
            serde_json::to_vec(&document).unwrap(),
            signing_key.verifying_key(),
        )
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()
    }

    #[test]
    fn verifies_valid_license() {
        let (source, key) = signed_license(json!({
            "customer": "customer",
            "deployment": "windows-portable",
            "issued_at": "2025-01-01T00:00:00Z",
            "expires_at": "2027-01-01T00:00:00Z"
        }));
        let verified = verify_license(&source, now(), &key).unwrap();
        assert_eq!(verified.info.customer, "customer");
    }

    #[test]
    fn rejects_tampered_payload() {
        let (source, key) = signed_license(json!({
            "customer": "customer",
            "deployment": "windows-portable",
            "expires_at": "2027-01-01T00:00:00Z"
        }));
        let mut document: Value = serde_json::from_slice(&source).unwrap();
        document["payload"]["customer"] = Value::String("other".to_string());
        let tampered = serde_json::to_vec(&document).unwrap();
        assert_eq!(
            verify_license(&tampered, now(), &key).unwrap_err(),
            "Invalid license signature."
        );
    }

    #[test]
    fn rejects_expired_license() {
        let (source, key) = signed_license(json!({
            "customer": "customer",
            "deployment": "docker",
            "expires_at": "2025-01-01T00:00:00Z"
        }));
        assert!(verify_license(&source, now(), &key)
            .unwrap_err()
            .starts_with("License expired at"));
    }

    #[test]
    fn rejects_missing_required_field() {
        let (source, key) = signed_license(json!({
            "customer": "customer",
            "expires_at": "2027-01-01T00:00:00Z"
        }));
        assert_eq!(
            verify_license(&source, now(), &key).unwrap_err(),
            "Invalid license payload: deployment is required."
        );
    }

    #[test]
    fn rejects_invalid_signature_base64() {
        let source = serde_json::to_vec(&json!({
            "payload": {
                "customer": "customer",
                "deployment": "docker",
                "expires_at": "2027-01-01T00:00:00Z"
            },
            "signature": "not base64"
        }))
        .unwrap();
        let key = SigningKey::from_bytes(&[7_u8; 32]).verifying_key();
        assert_eq!(
            verify_license(&source, now(), &key).unwrap_err(),
            "Invalid base64 value for signature."
        );
    }

    #[test]
    fn rejects_invalid_embedded_license_base64() {
        assert_eq!(
            decode_license_data_b64("not base64").unwrap_err(),
            "Invalid base64 value for TRAIN_PLATFORM_LICENSE_DATA_B64."
        );
    }

    #[test]
    fn rejects_missing_license_file() {
        let path = PathBuf::from("definitely-missing-license.dat");
        assert!(read_license_file(&path)
            .unwrap_err()
            .starts_with("License file not found:"));
    }

    #[test]
    fn accepts_naive_datetime_as_utc() {
        assert_eq!(
            parse_datetime("2027-01-01T00:00:00").unwrap(),
            Utc.with_ymd_and_hms(2027, 1, 1, 0, 0, 0).unwrap()
        );
    }

    #[test]
    fn canonical_json_matches_python_sort_keys() {
        let value = json!({"z": 1, "a": {"d": 2, "b": 1}});
        assert_eq!(
            String::from_utf8(canonical_payload_bytes(&value).unwrap()).unwrap(),
            r#"{"a":{"b":1,"d":2},"z":1}"#
        );
    }

    #[cfg(feature = "enforce-license")]
    #[test]
    fn release_feature_forces_license_requirement() {
        std::env::set_var("TRAIN_PLATFORM_LICENSE_REQUIRED", "0");
        assert!(license_required());
    }
}
