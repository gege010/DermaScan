import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { fileURLToPath } from 'node:url';
import assessment from './assessment.fixture.json' with { type: 'json' };

test('home specimen and navigation work without horizontal overflow', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: '02 Dermis' }).click();
  await expect(page.getByText('Lapisan penopang yang berisi')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  await page.getByRole('link', { name: 'Mulai kenali kulit' }).click();
  await expect(page.getByRole('heading', { name: 'Mulai dari ceritamu.' })).toBeVisible();
});

test('knowledge search, empty state and clear recover', async ({ page }) => {
  await page.goto('/panduan');
  await page.getByLabel('Cari panduan').fill('zzzzzzz');
  await expect(page.getByText('Belum menemukan yang kamu cari.')).toBeVisible();
  await page.getByRole('button', { name: 'Hapus pencarian' }).click();
  await expect(page.getByRole('status')).toContainText('6 panduan');
  await page.getByRole('button', { name: 'Kebiasaan harian' }).click();
  await expect(page.getByRole('status')).toContainText('1 panduan');
  await page.getByRole('link', { name: /Perlindungan matahari, setiap hari/ }).click();
  await expect(
    page.getByRole('heading', { name: 'Gabungkan beberapa bentuk perlindungan' }),
  ).toBeVisible();
});

test('journal create edit session navigation export and confirmed delete', async ({ page }) => {
  await page.goto('/jurnal');
  await page
    .getByLabel('Hal yang ingin kamu ingat')
    .fill('Catatan pengujian, bukan informasi pasien.');
  await page.getByRole('button', { name: 'Tambahkan catatan' }).click();
  await expect(page.getByText('Catatan pengujian, bukan informasi pasien.')).toBeVisible();
  await page.getByRole('button', { name: 'Sunting', exact: true }).click();
  await page.getByLabel('Hal yang ingin kamu ingat').fill('Catatan pengujian diperbarui.');
  await page.getByRole('button', { name: 'Simpan perubahan' }).click();
  await page.getByRole('link', { name: 'Kemampuan & privasi' }).click();
  await page.goBack();
  await expect(page.getByText('Catatan pengujian diperbarui.')).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Ekspor', exact: true }).click();
  expect((await download).suggestedFilename()).toBe('jurnal-kulit.txt');
  await page.getByRole('button', { name: /Hapus catatan/ }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('button', { name: 'Simpan catatan', exact: true }).click();
  await expect(page.getByText('Catatan pengujian diperbarui.')).toBeVisible();
  await page.getByRole('button', { name: /Hapus catatan/ }).click();
  await page
    .getByRole('dialog')
    .getByRole('button', { name: 'Hapus catatan', exact: true })
    .click();
  await expect(page.getByText('Cerita pertamamu')).toBeVisible();
});

test('journal persistence is explicit and removable', async ({ page }) => {
  await page.goto('/jurnal');
  await page
    .getByLabel('Hal yang ingin kamu ingat')
    .fill('Catatan sintetis untuk tes persistensi.');
  await page.getByRole('button', { name: 'Tambahkan catatan' }).click();
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
  await page.getByLabel('Simpan di perangkat ini').check();
  await page.reload();
  await expect(page.getByText('Catatan sintetis untuk tes persistensi.')).toBeVisible();
  await page.getByLabel('Simpan di perangkat ini').click();
  await page.getByRole('button', { name: 'Nonaktifkan', exact: true }).click();
  expect(await page.evaluate(() => localStorage.length)).toBe(0);
});

test('urgent direction is visible offline before any photo or API', async ({ page }) => {
  await page.goto('/analisis');
  await page.getByLabel('Sulit bernapas atau wajah membengkak').check();
  await expect(page.getByText('Jangan menunggu analisis.')).toBeVisible();
  await expect(page.getByText(/Cari pertolongan di IGD terdekat sekarang/)).toBeVisible();
  await page.getByRole('button', { name: 'Lihat langkah berikutnya' }).click();
  await expect(page.getByText('Centang persetujuan untuk melanjutkan.')).toBeVisible();
});

test('analysis failure preserves context and allows retry', async ({ page }) => {
  await page.route('**/api/v1/analyze', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ detail: 'Layanan sedang penuh. Coba lagi.' }),
    }),
  );
  await page.goto('/analisis');
  await page.getByLabel('Tahi lalat berubah atau luka tidak sembuh').check();
  await page.getByRole('checkbox', { name: /Saya setuju foto/ }).check();
  await page.getByRole('button', { name: 'Lihat langkah berikutnya' }).click();
  await expect(page.getByRole('alert')).toContainText('Layanan sedang penuh');
  await expect(page.getByLabel('Tahi lalat berubah atau luka tidak sembuh')).toBeChecked();
  await expect(page.getByRole('button', { name: 'Coba lagi', exact: true })).toBeEnabled();
});

test('accessible dialog keyboard cancellation restores focus', async ({ page }) => {
  await page.goto('/jurnal');
  await page.getByLabel('Hal yang ingin kamu ingat').fill('Catatan uji fokus.');
  await page.getByRole('button', { name: 'Tambahkan catatan' }).click();
  const remove = page.getByRole('button', { name: /Hapus catatan/ });
  await remove.click();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).not.toBeVisible();
  await expect(remove).toBeFocused();
});

for (const path of ['/', '/analisis', '/panduan', '/jurnal', '/pendekatan']) {
  test(`accessibility ${path}`, async ({ page }) => {
    await page.goto(path);
    await expect(page.getByRole('main')).toBeVisible();
    const audit = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze();
    expect(audit.violations).toEqual([]);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
  });
}

test('reduced motion is respected', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  expect(
    await page
      .locator('.skin-layer')
      .first()
      .evaluate((el) => getComputedStyle(el).transitionDuration),
  ).toBe('1e-05s');
});

test('photo preview, actual request state, honest result and journal handoff', async ({ page }) => {
  let finish!: () => void;
  const gate = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route('**/api/v1/analyze', async (route) => {
    await gate;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(assessment),
    });
  });
  await page.goto('/analisis');
  await page
    .locator('input[type=file]')
    .setInputFiles(fileURLToPath(new URL('./synthetic.png', import.meta.url)));
  await expect(page.getByAltText('Pratinjau foto kulit yang dipilih')).toBeVisible();
  await page.getByRole('checkbox', { name: /Saya setuju foto/ }).check();
  await page.getByRole('button', { name: 'Pahami kulitku' }).click();
  await expect(page.getByRole('button', { name: 'Memproses analisis' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Batalkan analisis' })).toBeVisible();
  finish();
  await expect(
    page.getByRole('heading', { name: 'Foto belum dapat diberi kesimpulan medis' }),
  ).toBeVisible();
  const audit = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
    .analyze();
  expect(audit.violations).toEqual([]);
  await page.getByRole('button', { name: 'Siapkan catatan jurnal' }).click();
  await expect(page.getByLabel('Hal yang ingin kamu ingat')).toHaveValue(
    /Foto belum dapat diberi kesimpulan medis/,
  );
});

test('visual proof and responsive navigation', async ({ page }, testInfo) => {
  await page.goto('/');
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: `../artifacts/dermascan-${testInfo.project.name}.png`,
    fullPage: false,
  });
  if (testInfo.project.name === 'mobile') {
    await page.getByRole('button', { name: 'Buka navigasi' }).click();
    await page
      .getByRole('navigation', { name: 'Navigasi seluler' })
      .getByRole('link', { name: 'Jurnal kulit' })
      .click();
    await expect(
      page.getByRole('heading', { name: 'Perubahan kecil, cerita yang berarti.' }),
    ).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Navigasi seluler' })).not.toBeVisible();
  }
});
