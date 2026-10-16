"""Admin view of the master catalog's Product Family > Tray hierarchy, as
mirrored into Postgres by backend/ingestion/seed_master_catalog.py.
Admin-only, same as the rest of catalog/document management (see
auth.dependencies.require_admin). The hierarchy itself -- which families and
trays exist, their catalog names, which tray sits under which family -- comes
from the master file and is read-only here; the one editable field is each
row's display name (issue #15), which a re-seed never overwrites.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException

from auth.dependencies import require_admin
from auth.repository import UserRecord
from catalog.models import DisplayNameUpdate, ProductFamilyOut, ProductFamilySummaryOut, TrayOut
from catalog.repository import (
    list_families_with_trays,
    set_family_display_name,
    set_tray_display_name,
)

router = APIRouter()


@router.get("/product-families", response_model=list[ProductFamilyOut])
async def list_product_families(
    admin: UserRecord = Depends(require_admin),
) -> list[ProductFamilyOut]:
    return [
        ProductFamilyOut(
            id=family.id,
            name=family.name,
            display_name=family.display_name,
            trays=[
                TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
                for tray in family.trays
            ],
        )
        for family in await list_families_with_trays()
    ]


@router.patch("/product-families/{family_id}", response_model=ProductFamilySummaryOut)
async def update_product_family_display_name(
    family_id: uuid.UUID,
    body: DisplayNameUpdate,
    admin: UserRecord = Depends(require_admin),
) -> ProductFamilySummaryOut:
    family = await set_family_display_name(family_id, body.display_name)
    if family is None:
        raise HTTPException(status_code=404, detail="product family not found")
    return ProductFamilySummaryOut(id=family.id, name=family.name, display_name=family.display_name)


@router.patch("/trays/{tray_id}", response_model=TrayOut)
async def update_tray_display_name(
    tray_id: uuid.UUID,
    body: DisplayNameUpdate,
    admin: UserRecord = Depends(require_admin),
) -> TrayOut:
    tray = await set_tray_display_name(tray_id, body.display_name)
    if tray is None:
        raise HTTPException(status_code=404, detail="tray not found")
    return TrayOut(id=tray.id, name=tray.name, display_name=tray.display_name)
